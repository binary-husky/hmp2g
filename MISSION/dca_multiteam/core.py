import numpy as np
import time
try:
    from numba import jit
except:
    from UTIL.tensor_ops import dummy_decorator as jit

# written by qth,2021/04/22
from scipy.optimize import linear_sum_assignment
from scipy.cluster.vq import kmeans2
from .cython_func import laser_hit_improve3
# action of the agent


# properties and state of physical world entity
class Entity(object):
    def __init__(self, size = 0.05 ,color = None):
        # properties:
        self.size = size
        # entity can move / be pushed
        self.movable = False
        # entity collides with others
        self.collide = True
        # material density (affects mass)
        self.density = 25.0
        # max speed and accel
        self.max_speed = None
        self.accel = None
        # state
        self.initial_mass = 1.0
        
    @property
    def mass(self):
        return self.initial_mass



# properties of agent entities
class Agent(Entity):
    def __init__(self, iden=None):
        super(Agent, self).__init__()
        # agents are movable by default
        self.movable = True
        # cannot send communication signals
        self.silent = False
        # cannot observe the world
        self.blind = False
        # physical motor noise amount
        self.u_noise = None
        # communication noise amount
        self.c_noise = None
        # control range
        self.u_range = 1.0
        # state
        self.act = None
        # script behavior to execute
        self.action_callback = None
        # script behavior to execute
        self.action_callback_test = None
        ## number of bullets hit
        self.numHit = 0         # overall
        self.numWasHit = 0
        self.hit = False        # in last time
        self.wasHit = False
        ## shooting cone's radius and width (in radian)
        self.shootRad = 0.4 
        self.shootWin = np.pi/4
        self.alive = True   # alive/dead
        self.justDied = False   # helps compute reward for agent when it just died
        self.prevDist = None
        self.uid = -1
        self.tid = -1
        if iden is not None:
            self.iden = iden

# multi-agent world
class World():
    def __init__(self):
        ## lists of agents, entities and bullets (can change at execution-time!)
        self.agents = []
        self.landmarks = []
        self.bullets = []
        # communication channel dimensionality
        self.dim_c = 0
        # position dimensionality
        self.dim_p = 3  ## x, y, angle
        # color dimensionality
        self.dim_color = 3
        # simulation timestep
        self.dt = 0.1
        # physical damping
        self.damping = 0.25
        # contact response parameters
        self.contact_force = 1e+2
        self.contact_margin = 1e-10 # 1e-3
        ## wall positions
        self.wall_pos = [-1, 1, -1, 1]  # (xmin, xmax) vertical and  (ymin,ymax) horizontal walls
        # written by qth, 2021/04/20,用于判断是否第一次初始化
        self.start_flag = True
        self.target_index = 0
        self.teams_result_step1 = None
        self.team_centroid_step1 = None
        from .collective_assult_parallel_run import ScenarioConfig
        self.s_cfg = ScenarioConfig


    # update state of the world
    def step(self):
        self.before_step_alive_agents = [a for a in self.agents if a.alive]
        ## -------- apply effects of laser ------------- ##
        self.apply_laser_effect()  
        
        # ------------- Calculate total physical (p_force) on each agent ------------- #
        p_force = [None] * len(self.before_step_alive_agents)
        # apply agent physical controls
        p_force = self.apply_action_force(p_force)
        # apply environment forces
        # p_force = self.apply_environment_force(p_force)
        ## apply wall collision forces
        p_force = self.apply_wall_collision_force(p_force)
        # integrate physical state
        # calculates new state based on forces
        self.integrate_state(p_force)

    # gather agent action forces
    def apply_action_force(self, p_force):
        for i,agent in enumerate(self.before_step_alive_agents):
            p_force[i] = agent.act[:2] 
        return p_force

    def apply_wall_collision_force(self, p_force):
        for a,agent in enumerate(self.before_step_alive_agents):
            f = self.get_wall_collision_force(agent)
            if(f is not None):
                assert p_force[a] is not None
                p_force[a] = f + p_force[a] 
        return p_force

    def set_terrain_adv(self):
        terrain_A = self.s_cfg.terrain_parameters[0]
        terrain_B = self.s_cfg.terrain_parameters[1]
        if self.s_cfg.introduce_terrain:
            pos_arr = np.array([a.pos for a in self.before_step_alive_agents])
            terrain = self.get_terrain(pos_arr, theta=self.init_theta, A=terrain_A, B=terrain_B)
            for i,entity in enumerate(self.before_step_alive_agents):
                entity.terrain = terrain[i]
        else:
            for i,entity in enumerate(self.before_step_alive_agents):
                entity.terrain = 1.0


    def apply_laser_effect(self):
        ## reset bullet hitting states
        for i,entity in enumerate(self.before_step_alive_agents):
            entity.hit = False
            entity.wasHit = False
            entity.wasHitBy = None

        for _, entity in enumerate(self.before_step_alive_agents):
            if entity.can_fire:
                for _, entity_b in enumerate(self.before_step_alive_agents):
                    if entity.team == entity_b.team: continue

                    fanRadius  = entity.shootRad*entity.terrain
                    fanOpenRad = entity.shootWin
                    fanDirRad  = entity.atk_rad
                    hit__4 = laser_hit_improve3(
                        entity.pos, entity_b.pos, 
                        fanRadius, fanOpenRad, fanDirRad
                    )
                    if hit__4:
                        base_prob = 0.8
                        terrain_advantage_delta_limit = 0.1
                        terrain_advantage = max(min(
                            entity.terrain - entity_b.terrain, 
                            terrain_advantage_delta_limit),
                            -terrain_advantage_delta_limit)
                        hit_prob_final = base_prob + terrain_advantage
                        if np.random.rand() < hit_prob_final:
                            entity.hit = True
                            entity.numHit += 1
                            entity_b.wasHit = True
                            entity_b.wasHitBy = entity
                            entity_b.numWasHit += 1

        # update just died state of dead agents
        for agent in self.agents:
            if not agent.alive:
                agent.justDied = False

        ## laser directly kills with one shot
        for agent in self.before_step_alive_agents:
            if agent.wasHit:
                agent.alive = False
                agent.justDied = True

    # integrate physical state
    def integrate_state(self, p_force):
        def reg_angle(rad):
            return (rad + np.pi)%(2*np.pi) -np.pi
        for i,entity in enumerate(self.before_step_alive_agents):
            if not entity.movable: continue
            entity.vel = entity.vel * (1 - self.damping)
            if (p_force[i] is not None):
                entity.vel += (p_force[i] / entity.mass) * self.dt
            if entity.max_speed is not None:
                speed = np.sqrt(np.square(entity.vel[0]) + np.square(entity.vel[1]))
                if speed > entity.max_speed:
                    entity.vel = entity.vel / np.sqrt(np.square(entity.vel[0]) +
                                                                    np.square(entity.vel[1])) * entity.max_speed
            ## simple model for rotation
            if entity.alive:
                entity.atk_rad += entity.act[2]
                entity.atk_rad = reg_angle(entity.atk_rad)
                
            entity.pos += entity.vel * self.dt


    # taking higher position, taking shoot range advantage
    @staticmethod
    @jit
    def get_terrain(arr, theta, A, B):
        # A = 0.05; B=0.2
        X=arr[:,0]; Y=arr[:,1]
        X_ = X*np.cos(theta) + Y*np.sin(theta)
        Y_ = -X*np.sin(theta) + Y*np.cos(theta)
        Z = -1 +B*( (0.1*X_) ** 2 + (0.1*Y_) ** 2 )- A * np.cos(2 * np.pi * (0.3*X_))  - A * np.cos(2 * np.pi * (0.5*Y_))
        return -Z


    #  fanRadius = agent.shootRad*agent.terrain
    #  fanOpenRad = agent.shootWin
    #  fanDirRad = agent.atk_rad
    def get_tri_pts_arr(self, agent):
        max_fire_range = agent.shootRad
        terrain = agent.terrain
        # assert terrain > 0.7 and terrain <= 1.2, (terrain, 'overflow')
        fire_range_fix = max_fire_range*terrain
        ang = agent.atk_rad
        # pt1 = agent.pos + agent.size*np.array([np.cos(ang), np.sin(ang)]) # 这句代码把发射点从中心点往atk_rad方向偏移一点
        pt1 = agent.pos # 去除掉，使得智能体的size不再影响攻击范围
        pt2 = pt1 + fire_range_fix*np.array([np.cos(ang+agent.shootWin/2), np.sin(ang+agent.shootWin/2)])
        pt3 = pt1 + fire_range_fix*np.array([np.cos(ang-agent.shootWin/2), np.sin(ang-agent.shootWin/2)])
        A = np.array([[pt1[0], pt2[0], pt3[0]],
                      [pt1[1], pt2[1], pt3[1]]])       
        return A, fire_range_fix





    # collision force with wall
    def get_wall_collision_force(self, entity):
        if not entity.collide_wall:
            return None # not a collider
        xmin,xmax,ymin,ymax = self.wall_pos
        x,y = entity.pos
        size = entity.size
        dists = np.array([x-(size+xmin), xmax-x-size, y-size-ymin, ymax-y-size])
        if (dists>0).all(): return np.array([0,0])

        # softmax penetration
        k = self.contact_margin
        penetration = np.logaddexp(0, -dists/k)*k
        fx1,fx2,fy1,fy2 = self.contact_force * penetration
        force = [fx1-fx2,fy1-fy2]
        return force