import pygame 
import numpy as np 
import time 
import tensorflow as tf
import statistics
import tqdm
import abc
import collections
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

	def _reset(self , s):
		self.ball = None 
		self.ball = Ball()
		self.paddle_a = PaddleA()

		self.paddle_b = PaddleB()

		# numpy_function

		if s== 'a':
			state , reward , done = np.array([self.paddle_a.y ,self.paddle_a.x- self.ball.x ,self.paddle_a.y-self.ball.y],dtype = np.int32),0,0

		if s=='b':
			state , reward , done = np.array([self.paddle_b.y ,self.paddle_b.x- self.ball.x ,self.paddle_b.y-self.ball.y] ,dtype = np.int32), 0 ,0

		return state , reward , done


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


	

		if s == 'a' and self.ball.reset == True:
			self.ball.reset = False
			self._current_time_step = ts.termination(np.array([self.paddle_a.y ,self.paddle_a.x- self.ball.x ,self.paddle_a.y-self.ball.y] , dtype = np.int32 ) , reward = -1000 )
			self._reset(s)
			return self._current_time_step

		if s == 'b' and self.ball.reset == True:

			self.ball.reset = False
			self._current_time_step = ts.termination(np.array([self.paddle_b.y ,self.paddle_b.x- self.ball.x ,self.paddle_b.y-self.ball.y] , dtype = np.int32 ) , reward = -1000 )
			self._reset(s)
			return self._current_time_step

		if s == 'a' and  (self.ball.x  -50 <= self.paddle_a.x and self.ball.y >= self.paddle_a.y and self.ball.y <= self.paddle_a.y+100) :
			 
			self._current_time_step = ts.transition(np.array([self.paddle_a.y ,self.paddle_a.x- self.ball.x ,self.paddle_a.y-self.ball.y] , dtype = np.int32) , reward = 1000, discount = 1)

		elif s == 'a' :
 			self._current_time_step = ts.transition(np.array([self.paddle_a.y ,self.paddle_a.x- self.ball.x ,self.paddle_a.y-self.ball.y] , dtype = np.int32) , reward = 1, discount = 1)

		elif s=='b' and (self.ball.x + 25 >= self.paddle_b.x and self.ball.y >= self.paddle_b.y and self.ball.y <= self.paddle_b.y+100):

			self._current_time_step = ts.transition(np.array([self.paddle_b.y ,self.paddle_b.x- self.ball.x ,self.paddle_b.y-self.ball.y] , dtype = np.int32) , reward = 1000, discount = 1)

		elif s == 'b' :
 			self._current_time_step = ts.transition(np.array([self.paddle_b.y ,self.paddle_b.x- self.ball.x ,self.paddle_b.y-self.ball.y] , dtype = np.int32) , reward = 1, discount = 1)


		return self._current_time_step


	def action_spec(self):

		return self._action_spec

	def observation_spec(self):

		return self._observation_spec


py_env = Ping_Pong_Env('a')
# py_env= tf_py_environment.TFPyEnvironment(py_env)



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

def env_step(action: np.ndarray ,s):
	"""Returns state, reward and done flag given an action."""

	state, reward, done = py_env.step(action,s)
	return (state.astype(np.float32), np.array(reward, np.int32), np.array(done, np.int32))


def tf_env_step(action: tf.Tensor ,s):
	print('tf.numpy_function(env_step, [action , s], [tf.float32, tf.int32, tf.int32])',tf.numpy_function(env_step, [action , s], [tf.float32, tf.int32, tf.int32]))
	return tf.numpy_function(env_step, [action , s], [tf.float32, tf.int32, tf.int32])

def env_reset(s):
	state , reward , done = py_env._reset(s)
	return (state.astype(np.int32), np.array(reward, np.int32), np.array(done, np.int32))

def tf_env_reset(s):
	# print('tf.numpy_function(/, [action , s], [tf.float32, tf.int32, tf.int32])',tf.numpy_function(env_reset, [ s], [tf.float32, tf.int32, tf.int32]))
	print(env_reset(s))
	state , reward , done = env_reset(s)
	# return tf.numpy_function(env_reset,[s], [tf.int32, tf.int32, tf.int32])
	return [state , reward , done]


def run_episode(initial_state,  modelA , modelB, max_steps , s) :

	action_probs_a = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
	values_a = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
	rewards_a = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)
	action_probs_b = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
	values_b = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
	rewards_b = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)


	initial_state_shape_a = initial_state_a.shape
	state_a = initial_state_a
	initial_state_shape_b = initial_state_b.shape
	state_b = initial_state_b

	for t in tf.range(max_steps):
		# Convert state into a batched tensor (batch size = 1)
		state_a = tf.expand_dims(state_a, 0)

		# Run the model and to get action probabilities and critic value
		action_logits_t_a, value_a = modelA(state_a)

		state_b = tf.expand_dims(state_b, 0)
		action_logits_t_b, value_b = modelB(state_b)

		# Sample next action from the action probability distribution
		action_a = tf.random.categorical(action_logits_t_a, 1)[0, 0]
		action_probs_t_a = tf.nn.softmax(action_logits_t_a)

		action_b = tf.random.categorical(action_logits_t_b, 1)[0, 0]
		action_probs_t_b = tf.nn.softmax(action_logits_t_b)

		# Store critic values
		values_a = values_a.write(t, tf.squeeze(value_a))

		# Store log probability of the action chosen
		action_probs_a = action_probs_a.write(t, action_probs_t_a[0, action])

		# Store critic values
		values_b = values_b.write(t, tf.squeeze(value_b))

		# Store log probability of the action chosen
		action_probs_b = action_probs_b.write(t, action_probs_t_b[0, action])

		# Apply action to the environment to get next state and reward
		state_a, reward_a, done_a = tf_env_step(action_a,'a')
		state_a.set_shape(initial_state_shape_a)

		state_b, reward_b, done_b = tf_env_step(action_b,'b')
		state_b.set_shape(initial_state_shape_b)

		# Store reward
		rewards_a = rewards_a.write(t, reward_a)
		rewards_b = rewards_b.write(t, reward_b)

		if tf.cast(done_a, tf.bool):
			break
		if tf.cast(done_b, tf.bool):
			break

	action_probs_a = action_probs_a.stack()
	values_a = values_a.stack()
	rewards_a = rewards_a.stack()

	action_probs_b = action_probs_b.stack()
	values_b = values_b.stack()
	rewards_b = rewards_b.stack()

	return action_probs_a, values_a, rewards_a ,action_probs_b, values_b, rewards_b

optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
def get_expected_return(rewards, gamma, standardize = True ):
	"""Compute expected returns per timestep."""

	n = tf.shape(rewards)[0]
	returns = tf.TensorArray(dtype=tf.float32, size=n)

	# Start from the end of `rewards` and accumulate reward sums
	# into the `returns` array
	rewards = tf.cast(rewards[::-1], dtype=tf.float32)
	discounted_sum = tf.constant(0.0)
	discounted_sum_shape = discounted_sum.shape
	for i in tf.range(n):
		reward = rewards[i]
		discounted_sum = reward + gamma * discounted_sum
		discounted_sum.set_shape(discounted_sum_shape)
		returns = returns.write(i, discounted_sum)
	returns = returns.stack()[::-1]

	if standardize:
		returns = ((returns - tf.math.reduce_mean(returns)) / (tf.math.reduce_std(returns) + eps))

	return returns

def compute_loss(action_probs,  values,returns):
	"""Computes the combined actor-critic loss."""

	advantage = returns - values

	action_log_probs = tf.math.log(action_probs)
	actor_loss = -tf.math.reduce_sum(action_log_probs * advantage)

	critic_loss = huber_loss(values, returns)

	return actor_loss + critic_loss

@tf.function
def train_step(initial_state_a,initial_state_b, modelA , modelB, optimizer, gamma, max_steps_per_episode):
	"""Runs a model training step."""

	with tf.GradientTape() as tape:

		# Run the model for one episode to collect training data
		action_probs_a, values_a, rewards_a ,action_probs_b, values_b, rewards_b = run_episode(initial_state_a , initial_state_b, modelA ,modelB, max_steps_per_episode) 

		# Calculate expected returns
		returns_a = get_expected_return(rewards_a, gamma)
		returns_b = get_expected_return(rewards_b, gamma)

		# Convert training data to appropriate TF tensor shapes
		action_probs, values, returns = [ tf.expand_dims(x, 1) for x in [action_probs, values, returns]] 

		# Calculating loss values to update our network
		loss_a = compute_loss(action_probs_a, values_a, returns_a)
		loss_b = compute_loss(action_probs_b, values_b, returns_b)

		# Compute the gradients from the loss
	grads_a = tape.gradient(loss_a, modelA.trainable_variables)

	# Apply the gradients to the model's parameters
	optimizer.apply_gradients(zip(grads_a, modelA.trainable_variables))

	grads_b = tape.gradient(loss_b, modelA.trainable_variables)

	# Apply the gradients to the model's parameters
	optimizer.apply_gradients(zip(grads_b, modelB.trainable_variables))

	episode_reward_a = tf.math.reduce_sum(rewards_a)
	episode_reward_b = tf.math.reduce_sum(rewards_b)

	return episode_reward_a ,episode_reward_b





min_episodes_criterion = 100
max_episodes = 10000
max_steps_per_episode = 1000

# Cartpole-v0 is considered solved if average reward is >= 195 over 100 
# consecutive trials
reward_threshold = 195
running_reward = 0

# Discount factor for future rewards
gamma = 0.99

# Keep last episodes reward
episodes_rewardA: collections.deque = collections.deque(maxlen=min_episodes_criterion)
episodes_rewardB: collections.deque = collections.deque(maxlen=min_episodes_criterion)

with tqdm.trange(max_episodes) as t:
	for i in t:
		initial_state_a = tf.ragged.constant(tf_env_reset('a'), dtype=tf.float32)
		initial_state_b = tf.ragged.constant(tf_env_reset('b'), dtype=tf.float32)

		episode_reward_a , episode_reward_b = int(train_step(
		    initial_state_a,initial_state_b, modelA , modelB, optimizer, gamma, max_steps_per_episode))

		episodes_rewardA.append(episode_reward_a)
		running_reward_a = statistics.mean(episodes_rewardA)
		episodes_rewardB.append(episode_reward_b)
		running_reward_B = statistics.mean(episodes_rewardB)

		t.set_description(f'Episode {i}')
		t.set_postfix(episode_reward=episode_rewardA, running_reward=running_reward_a)
		t.set_postfix(episode_reward=episode_rewardB, running_reward=running_reward_b)

		# Show average episode reward every 10 episodes
		if i % 10 == 0:
			pass # print(f'Episode {i}: average reward: {avg_reward}')

		if running_reward_a > reward_threshold and running_reward_a > reward_threshold and i >= min_episodes_criterion:  
			break


print(f'\nSolved at episode {i}: average reward: {running_reward:.2f}!')






















	
