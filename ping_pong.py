import pygame 
import numpy as np 
import time 
import tensorflow as tf
from tf_agents.specs import array_spec , tensor_spec
from tf_agents.trajectories import time_step as ts 
from tf_agents.environments import tf_py_environment
from tensorflow.keras import layers
from tf_agents.environments import py_environment
WIN = pygame.display.set_mode((600,800))
pygame.font.init()
STAT_FONT = pygame.font.SysFont("comicsans" , 50)
class PaddleA:

	def __init__(self):

		self.x = 0
		self.y = 0
		self.score =0

	def reset(self):
		self.x = 0
		self.y = 0
		self.score =0


	def draw(self,Win):
		pygame.draw.rect(Win , (255,0,0) ,  pygame.Rect(self.x,self.y, 30, 100))
		

	def move(self , vel):
		if vel<0 and self.y>20:
			self.y+=vel
		elif vel>0 and self.y<700 :
			self.y+= vel

class PaddleB:

	def __init__(self):

		self.x = 570
		self.y = 0
		self.score =0

	def reset(self):
		self.x = 570
		self.y = 0
		self.score =0

	def draw(self,Win):
		pygame.draw.rect(Win , (255,0,0) ,  pygame.Rect(self.x,self.y, 30, 100))
		

	def move(self , vel):
		if vel<0 and self.y>20:
			self.y+=vel
		elif vel>0 and self.y<700 :
			self.y+= vel


class Ball:

	def __init__(self):

		self.x = 300
		self.y = 400
		self.x_vel = 2
		self.y_vel = 2
		self.reset  = False 

	def reset(self):

		self.x = 300
		self.y = 400
		self.x_vel = 2
		self.y_vel = 2


	def draw(self,win):

		pygame.draw.circle(win , (0,0,255),(self.x,self.y)  ,25)

	def move(self,paddle_a , paddle_b):

		if self.y <=0 or self.y>=800 :
			self.y_vel = -self.y_vel

		if self.x <=0 or self.x>=600 :
			self.reset = True

		if self.x  -50 <= paddle_a.x and self.y >= paddle_a.y and self.y <= paddle_a.y+100:
			 self.x_vel = -self.x_vel  +np.random.randint(-1, 1) 
			 self.y_vel = -self.y_vel +np.random.randint(-1, 1) 
			 paddle_a.score+=1

		if self.x + 25 >= paddle_b.x and self.y >= paddle_b.y and self.y <= paddle_b.y+100:
			 self.x_vel = -self.x_vel +np.random.randint(-1, 1) 
			 self.y_vel = -self.y_vel +np.random.randint(-1, 1) 
			 paddle_b.score+=1
			 	 
		self.x+=self.x_vel
		self.y+= self.y_vel

	




paddle_a = PaddleA()
paddle_b = PaddleB()
ball = Ball()
def reset():
	paddle_a.reset()
	paddle_b.reset()
	ball.reset()
def draw_window():
	Win = pygame.display.set_mode((600,800))
	Win.fill((0,0,0))
	paddle_a.draw(Win)
	paddle_b.draw(Win)
	ball.draw(Win)
	score_label_A = STAT_FONT.render('Score PaddleA : '+str(paddle_a.score) , 1,(255,255,255))
	Win.blit(score_label_A , (600 - score_label_A.get_width()-400 , 40))
	score_label_B = STAT_FONT.render('Score PaddleB : '+str(paddle_b.score) , 1,(255,255,255))
	Win.blit(score_label_B , (600 - score_label_B.get_width()-400 , 80))

	pygame.display.update()

# while True:
# 	for event in pygame.event.get():
# 			if event.type == pygame.QUIT:
				
# 				pygame.quit()
# 				quit()
# 				break

# 			if event.type == pygame.KEYDOWN:
# 				if event.key == pygame.K_UP:
# 					paddle_a.move(-100)
# 				if event.key == pygame.K_DOWN:
# 					paddle_a.move(100)
# 				if event.key == pygame.K_w:
# 					paddle_b.move(-100)
# 				if event.key == pygame.K_s:
# 					paddle_b.move(100)

# 			if ball.reset == True:
# 				break
									
# 	draw_window()
# 	ball.move(paddle_a , paddle_b)


class Ping_Pong_Env(py_environment.PyEnvironment):

	def __init__(self , Str):
		self.ball = Ball()
		self.Str  = Str

		self.paddle_a = PaddleA()


		self.paddle_b= PaddleB()
		self._action_spec = array_spec.BoundedArraySpec(shape = () , dtype =np.float32 , minimum = 0 ,maximum = 1 , name = 'action')
		self._observation_spec = array_spec.BoundedArraySpec(shape=(6,) , dtype = np.int32 , name = 'observation' )
		self._episode_ended = False

	def render(self):
		win = pygame.display.set_mode((600,800))
		win.fill((0,0,0))
		self.ball.draw(win)
		self.paddle_a.draw(win)
		self.paddle_b.draw(win)

	def _reset(self):
		self.ball = None 
		self.ball = Ball()
		self.paddle_a = PaddleA()

		self.paddle_b = PaddleB()

		self._current_time_step = ts.restart(np.array([self.paddle_a.y ,self.paddle_a.x- self.ball.x ,self.paddle_a.y-self.ball.y,self.paddle_b.y ,self.paddle_b.x- self.ball.x ,self.paddle_b.y-self.ball.y] ,dtype = np.int32 ))
		return self._current_time_step

	def _step(self,action ,s):

		if s == 'a':

			if action <0.5:
				paddle_a.move(100)

			elif action == 0:
				paddle_a.move(-100)

		if s == 'b':

			if action <=0.5:
				paddle_b.move(100)

			elif action == 0:
				paddle_b.move(-100)


	

		if self.ball.reset == True:
			self.ball.reset = False
			self._current_time_step = ts.termination(np.array([self.paddle_a.y ,self.paddle_a.x- self.ball.x ,self.paddle_a.y-self.ball.y,self.paddle_b.y ,self.paddle_b.x- self.ball.x ,self.paddle_b.y-self.ball.y] , dtype = np.int32 ) , reward = -1000 )
			self._reset()
			return self._current_time_step

		if (self.ball.x  -50 <= self.paddle_a.x and self.ball.y >= self.paddle_a.y and self.ball.y <= self.paddle_a.y+100) or(self.ball.x + 25 >= self.paddle_b.x and self.ball.y >= self.paddle_b.y and self.ball.y <= self.paddle_b.y+100):
			 
			self._current_time_step = ts.transition(np.array([self.paddle_a.y ,self.paddle_a.x- self.ball.x ,self.paddle_a.y-self.ball.y,self.paddle_b.y ,self.paddle_b.x- self.ball.x ,self.paddle_b.y-self.ball.y] , dtype = np.int32) , reward = 1000, discount = 1)

		else :
 			self._current_time_step = ts.transition(np.array([self.paddle_a.y ,self.paddle_a.x- self.ball.x ,self.paddle_a.y-self.ball.y,self.paddle_b.y ,self.paddle_b.x- self.ball.x ,self.paddle_b.y-self.ball.y] , dtype = np.int32) , reward = 1, discount = 1)

		
		return self._current_time_step


	def action_spec(self):

		return self._action_spec

	def observation_spec(self):

		return self._observation_spec


py_env = Ping_Pong_Env('a')
py_env= tf_py_environment.TFPyEnvironment(py_env)



class Actor_Critic_A(tf.keras.Model):

	def __init__(self , num_actions , num_hidden_units):

		super().__init__()

		self.common = layers.Dense(num_hidden_units ,activation = 'relu')
		self.actor = layers.Dense(num_actions , activation = 'softmax')
		self.critic = layers.Dense(1)

	def call(self,inputs):

		x = self.common(inputs)
		return self.actor(x) , self.critic(x)


num_actions = 2
num_hidden_units = 128

modelA = Actor_Critic_A(num_actions, num_hidden_units)
modelB = Actor_Critic_A(num_actions , num_hidden_units)

def run_episode(initial_state,  model, max_steps) :

  action_probs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
  values = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
  rewards = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)

  initial_state_shape = initial_state.shape
  state = initial_state

  for t in tf.range(max_steps):
    # Convert state into a batched tensor (batch size = 1)
    state = tf.expand_dims(state, 0)

    # Run the model and to get action probabilities and critic value
    action_logits_t, value = model(state)

    # Sample next action from the action probability distribution
    action = tf.random.categorical(action_logits_t, 1)[0, 0]
    action_probs_t = tf.nn.softmax(action_logits_t)

    # Store critic values
    values = values.write(t, tf.squeeze(value))

    # Store log probability of the action chosen
    action_probs = action_probs.write(t, action_probs_t[0, action])

    # Apply action to the environment to get next state and reward
    state, reward, done = tf_env_step(action)
    state.set_shape(initial_state_shape)

    # Store reward
    rewards = rewards.write(t, reward)

    if tf.cast(done, tf.bool):
      break

  action_probs = action_probs.stack()
  values = values.stack()
  rewards = rewards.stack()

  return action_probs, values, rewards





























	