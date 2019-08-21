import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
from os import path
from gym.envs.classic_control import rendering

class MartySwingEnv(gym.Env):
    metadata = {
        'render.modes' : ['human', 'rgb_array'],
        'video.frames_per_second' : 30
    }

    def __init__(self, g=9.81, l1=0.45, l2=0.42, m=0.3, thetaInitDeg=30, vInitial=0):
        # Marty mass
        self.m = m

        # Time increment and acc-due-to-gravity for sim
        self.dt = .05
        self.g = g

        # Equivalent length when extended and kicked (initially extended)
        self.l1 = l1
        self.l2 = l2
        self.l = self.l1

        # Initial angle to vertical (anti-clockwise from vertically downwards)
        self.thetaInit = np.radians(thetaInitDeg)

        
        # Initial velocity
        self.vInitial = vInitial

        # Calculate system energies from theta and v
        self.basePotentialE = self.l1 * self.m * self.g
        self.kineticE = self.calcKineticEnergy(self.vInitial)
        self.potentialE = self.calcPotentialEnergy(self.thetaInit)

        # Viewer
        self.viewer = None
        self.lenRope = 1
        self.lenBody = 0.7
        self.lenLegTop = 0.4
        self.lenLegBottom = 0.4
        self.kickAngleKicked =  np.radians(20)
        self.kickAngleStright = 0

        # Action space - kick, straighten
        self.action_space = spaces.Discrete(2)

        # Observation space - X accelerometer reading only
        self.maxXAcc = self.g
        high = np.array([self.maxXAcc])
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

        # Reset
        self.reset()

        # Randomness
        self.seed()

    def reset(self):
        # Time elapsed
        self.t = 0

        # Angles
        self.theta = self.thetaInit
        self.thetaMax = self.theta
        self.kickAngle = self.kickAngleStright

        # Tangential velocity
        self.v = self.vInitial

        # Rewards and completion
        self.thetaPeakCount = 0

        return self._get_obs()

    def calcPotentialEnergy(self, theta):
        return self.m * self.g * (self.l1 - np.cos(theta) * self.l)

    def calcKineticEnergy(self, v):
        return 0.5 * self.m * v * v

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, act):
        # Immediate implementation of action
        reward = 0
        if act == 0:
            # Kick if we are not already in a kick state
            if self.kickAngle != self.kickAngleKicked:
                self.l = self.l2
                self.kickAngle = self.kickAngleKicked
                reward -= 0.1
        else:
            # Strighten if we are not already in a kick state
            if self.kickAngle != self.kickAngleStright:
                self.l = self.l1
                self.kickAngle = self.kickAngleStright
                reward -= 0.03

        # Update tangential velocity based on acceleration
        tangentialAcc = self.g * np.sin(self.theta)
        newV = self.v - tangentialAcc * self.dt

        # Calculate arc-angle traversed at current v in time dt
        arcLen = newV * self.dt
        thetaDiff = arcLen / self.l
        newTheta = self.theta + thetaDiff

        # Check for change of direction
        done = False
        if (newV > 0) != (self.v > 0):
            self.thetaPeakCount += 1
            if self.thetaPeakCount > 20:
                done = True
            if self.thetaMax < abs(newTheta):
                # reward += (abs(newTheta) - self.thetaMax) * 1000
                self.thetaMax = abs(newTheta)
            reward += abs(newTheta) * 10

        # Potential energy change
        newPotentialE = self.calcPotentialEnergy(newTheta)

        # Kinetic
        newKineticE = self.calcKineticEnergy(newV)

        # Update the state
        self.v = newV
        self.kineticE = newKineticE
        self.potentialE = newPotentialE
        self.theta = newTheta
        self.t += self.dt
        return self._get_obs(), reward, done, {"t":self.t, "PE":self.potentialE, "KE":self.kineticE, "v":self.v, "l":self.l, "theta":self.theta, "kickAngle":self.kickAngle, "thetaMax":self.thetaMax}

    def _get_obs(self):
        xAcc = - self.g * np.sin(self.theta)
        return np.array([xAcc])

    def makeRect(self, length, width):
        l, r, t, b = 0, length, width/2, -width/2
        return rendering.make_polygon([(l,b), (l,t), (r,t), (r,b)])

    def render(self, mode='human'):

        if self.viewer is None:

            # First time through we create the scene 
            self.viewer = rendering.Viewer(500,500)
            self.viewer.set_bounds(-2.2,2.2,-3.4,1.0)

            # Top axis
            axle = rendering.make_circle(.05)
            axle.set_color(0,0,0)
            self.viewer.add_geom(axle)

            # Rope suspending Marty
            rope = self.makeRect(self.lenRope, .02)
            rope.set_color(.2, .2, .2)
            self.ropeTransform = rendering.Transform()
            rope.add_attr(self.ropeTransform)
            self.viewer.add_geom(rope)

            # Body of the robot
            body = self.makeRect(self.lenBody, .5)
            body.set_color(10/255, 141/255, 231/255)
            self.bodyTransform = rendering.Transform()
            body.add_attr(self.bodyTransform)
            self.viewer.add_geom(body)

            # Leg parts
            legTop = self.makeRect(self.lenLegTop, .3)
            legTop.set_color(28/255, 173/255, 252/255)
            self.legTopTransform = rendering.Transform()
            legTop.add_attr(self.legTopTransform)
            self.viewer.add_geom(legTop)

            legBottom = self.makeRect(self.lenLegBottom, .3)
            legBottom.set_color(28/255, 173/255, 252/255)
            self.legBottomTransform = rendering.Transform()
            legBottom.add_attr(self.legBottomTransform)
            self.viewer.add_geom(legBottom)

        # Transformations to display the rope and Marty as he swings
        self.ropeTransform.set_rotation(self.theta - np.pi/2)
        martyTopPos = (np.cos(self.theta - np.pi/2)*self.lenRope, np.sin(self.theta - np.pi/2)*self.lenRope)
        self.bodyTransform.set_translation(martyTopPos[0], martyTopPos[1])
        self.bodyTransform.set_rotation(self.theta - np.pi/2)

        # Transformations to display Marty's legs as they kick
        legTopPos = (martyTopPos[0] + np.cos(self.theta - np.pi/2)*self.lenBody, martyTopPos[1] + np.sin(self.theta - np.pi/2)*self.lenBody)
        self.legTopTransform.set_translation(legTopPos[0], legTopPos[1])
        self.legTopTransform.set_rotation(self.theta - self.kickAngle - np.pi/2)
        legBottomPos = (legTopPos[0] + np.cos(self.theta - self.kickAngle - np.pi/2)*self.lenLegTop, 
                                legTopPos[1] + np.sin(self.theta - self.kickAngle - np.pi/2)*self.lenLegTop)
        self.legBottomTransform.set_translation(legBottomPos[0], legBottomPos[1])
        self.legBottomTransform.set_rotation(self.theta - np.pi/2)

        # Render
        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

