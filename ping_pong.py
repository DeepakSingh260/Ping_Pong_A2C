import pygame 
import numpy as np 
import time 
import tensorflow as tf
import statistics
import tqdm
import abc
import collections
from tf_agents.specs import array_spec 
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import time_step as ts 
from tf_agents.environments import tf_py_environment
from tensorflow.keras import layers
from tf_agents.environments import py_environment
eps = np.finfo(np.float32).eps.item()
WIN = pygame.display.set_mode((600,800))
pygame.font.init()
STAT_FONT = pygame.font.SysFont("comicsans" , 50)
clock = pygame.time.Clock()
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
		self.x_vel = 7
		self.y_vel = 7
		self.reset_a  = False
		self.reset_b = False 

	def reset(self):

		self.x = 300
		self.y = 400
		self.x_vel = 7
		self.y_vel = 7


	def draw(self,win):

		pygame.draw.circle(win , (0,0,255),(self.x,self.y)  ,25)

	def move(self,paddle_a , paddle_b):

		if self.y <=0 or self.y>=800 :
			self.y_vel = -self.y_vel

		if self.x <=0  :
			 self.reset_a = True
			 print('reset A set True ')
			 self.x_vel = abs(self.x_vel)  +np.random.randint(-1, 1) 
			 self.y_vel = -self.y_vel +np.random.randint(-1, 1) 


		if  self.x>=600:
			 print("Reset B set True")
			 self.reset_b = True
			 self.x_vel = -abs(self.x_vel) +np.random.randint(-1, 1) 
			 self.y_vel = -self.y_vel +np.random.randint(-1, 1) 


		if self.x  -50 <= paddle_a.x and self.y >= paddle_a.y and self.y <= paddle_a.y+100:
			 self.x_vel = abs(self.x_vel)  +np.random.randint(-1, 1) 
			 self.y_vel = -self.y_vel +np.random.randint(-1, 1) 
			 paddle_a.score+=1

		if self.x + 25 >= paddle_b.x and self.y >= paddle_b.y and self.y <= paddle_b.y+100:
			 self.x_vel = -abs(self.x_vel) +np.random.randint(-1, 1) 
			 self.y_vel = -self.y_vel +np.random.randint(-1, 1) 
			 paddle_b.score+=1
			 	 
		self.x+=self.x_vel
		self.y+= self.y_vel

	




paddle_a = PaddleA()
paddle_b = PaddleB()
ball = Ball()

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
		clock.tick(30)
		# print('rendering')
		win = pygame.display.set_mode((600,800))
		win.fill((0,0,0))
		self.ball.draw(win)
		self.paddle_a.draw(win)
		self.paddle_b.draw(win)
		score_label_A = STAT_FONT.render('Score PaddleA : '+str(self.paddle_a.score) , 1,(255,255,255))
		win.blit(score_label_A , (600 - score_label_A.get_width()-400 , 40))
		score_label_B = STAT_FONT.render('Score PaddleB : '+str(self.paddle_b.score) , 1,(255,255,255))
		win.blit(score_label_B , (600 - score_label_B.get_width()-400 , 80))


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
		action = tf.cast(action, tf.float32)
		# print('action' , action , 's' , s)
		self.ball.move(self.paddle_a  ,self.paddle_b)
		if s == 'a':

			if action >0.5:
				# print('a>0.5.......................................................................')
				self.paddle_a.move(100)

			else:
				# print('a<0.5.......................................................................')

				self.paddle_a.move(-100)

		if s == 'b':

			if action >0.5:
				# print('b>0.5.......................................................................')

				self.paddle_b.move(100)

			else:
				# print('b<0.5.......................................................................')

				self.paddle_b.move(-100)


	

		if  self.ball.reset_a == True:
			self.ball.reset_b = False
			state , rewrard , done  = np.array([self.paddle_a.y ,self.paddle_a.x- self.ball.x ,self.paddle_a.y-self.ball.y] , dtype = np.int32 ) , -((abs(self.paddle_a.x- self.ball.x)*abs(self.paddle_a.y-self.ball.y))//10 ),True
			# self._reset(s)
			return state , rewrard ,done 

		if self.ball.reset_b == True:

			self.ball.reset_b = False
			state , rewrard , done  = np.array([self.paddle_b.y ,self.paddle_b.x- self.ball.x ,self.paddle_b.y-self.ball.y] , dtype = np.int32 ) ,-((abs(self.paddle_b.x- self.ball.x)*abs(self.paddle_b.y-self.ball.y)) //10),True
			# self._reset(s)
			return state , rewrard , done 

		if s == 'a' and  (self.ball.x - 50 <= self.paddle_a.x and self.ball.y >= self.paddle_a.y and self.ball.y <= self.paddle_a.y+100) :
			 
			state , rewrard , done  = np.array([self.paddle_a.y ,self.paddle_a.x- self.ball.x ,self.paddle_a.y-self.ball.y] , dtype = np.int32) , 1000, False

		elif s == 'a' :
 			state , rewrard , done = np.array([self.paddle_a.y ,self.paddle_a.x- self.ball.x ,self.paddle_a.y-self.ball.y] , dtype = np.int32) , 10, False

		elif s=='b' and (self.ball.x + 25 >= self.paddle_b.x and self.ball.y >= self.paddle_b.y and self.ball.y <= self.paddle_b.y+100):

			state , rewrard , done  = np.array([self.paddle_b.y ,self.paddle_b.x- self.ball.x ,self.paddle_b.y-self.ball.y] , dtype = np.int32) ,1000, False

		elif s == 'b' :
 			state , rewrard , done  = np.array([self.paddle_b.y ,self.paddle_b.x- self.ball.x ,self.paddle_b.y-self.ball.y] , dtype = np.int32) , 10, False


		return 	 state, rewrard , done 


	def action_spec(self):

		return self._action_spec

	def observation_spec(self):

		return self._observation_spec


py_env = Ping_Pong_Env('a')
# py_env= tf_py_environment.TFPyEnvironment(py_env)



class Actor_Critic_A(tf.keras.Model):

	def __init__(self , num_actions , num_hidden_units):

		super().__init__()	
		# self.Input  = tf.keras.layers.InputLayer(input_shape=(3,3))
		# self.Flatten = tf.keras.layers.Flatten()

		self.common = layers.Dense(num_hidden_units ,activation = 'relu')
		self.actor = layers.Dense(num_actions , activation = 'softmax')
		self.critic = layers.Dense(1)

	def call(self,inputs):
		# inputs = self.Input(inputs)
		# inputs = self.Flatten(inputs)
		x = self.common(inputs)
		return self.actor(x) , self.critic(x)


num_actions = 2
num_hidden_units = 256

modelA = Actor_Critic_A(num_actions, num_hidden_units)
modelB = Actor_Critic_A(num_actions , num_hidden_units)

def env_step(action ,s):
	"""Returns state, reward and done flag given an action."""

	state, reward, done = py_env._step(action,s)
	return (state.astype(np.float32), np.array(reward, np.int32), np.array(done, np.int32))


def tf_env_step(action ,s):
	# print('tf.numpy_function(env_step, [action , s], [tf.float32, tf.int32, tf.int32])',tf.numpy_function(env_step, [action , s], [tf.float32, tf.int32, tf.int32]))
	return tf.numpy_function(env_step, [action , s], [tf.float32, tf.int32, tf.int32])

def env_reset(s):
	state , reward , done = py_env._reset(s)
	return (state.astype(np.int32), np.array([reward], np.int32), np.array([done], np.int32))

def tf_env_reset(s):
	# print('tf.numpy_function(/, [action , s], [tf.float32, tf.int32, tf.int32])',tf.numpy_function(env_reset, [ s], [tf.float32, tf.int32, tf.int32]))
	# print(env_reset(s))

	state , reward , done = env_reset(s)
	# return tf.numpy_function(env_reset,[s], [tf.int32, tf.int32, tf.bool])
	return np.array([state])  


def run_episode(initial_state_a , initial_state_b,  modelA , modelB, max_steps ) :

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

	# print(initial_state_a, 'initial_state_a')
	First_Bool = False
	Second_Bool  = False

	for t in tf.range(max_steps):
		# Convert state into a batched tensor (batch size = 1)
		# print('t..............................................',t)
		state_a = tf.expand_dims(state_a, 0)
		# print('state_a' , state_a.shape)

		# Run the model and to get action probabilities and critic value
		action_logits_t_a, value_a = modelA(state_a)

		state_b = tf.expand_dims(state_b, 0)
		action_logits_t_b, value_b = modelB(state_b)
		# print('logit',action_logits_t_a , 'first index' , action_logits_t_a[0])
		# print('logit shape' , action_logits_t_a.shape)
		if action_logits_t_a.shape == (1,1,2):
			# print('action_logits_t_a',tf.random.categorical(action_logits_t_a[0],1))

			# Sample next action from the action probability distribution
			action_a = tf.random.categorical(action_logits_t_a[0], 1)[0,0]
			# action_a = action_logits_t_a[0][0][0]

			action_probs_t_a = tf.nn.softmax(action_logits_t_a[0])
			# print('action_probs_t_a' , action_probs_t_a)

			action_b = tf.random.categorical(action_logits_t_b[0], 1 )[0,0]
			# action_b = action_logits_t_b[0][0][0]
			action_probs_t_b = tf.nn.softmax(action_logits_t_b[0])

		else:
			# print('action_logits_t_a',tf.random.categorical(action_logits_t_a[0],1))

			# Sample next action from the action probability distribution
			action_a = tf.random.categorical(action_logits_t_a, 1)[0,0]
			# action_a = action_logits_t_a[0][0]
			action_probs_t_a = tf.nn.softmax(action_logits_t_a)
			# print('action_probs_t_a' , action_probs_t_a)

			action_b = tf.random.categorical(action_logits_t_b, 1)[0,0]
			# action_b = action_logits_t_b[0][0]
			action_probs_t_b = tf.nn.softmax(action_logits_t_b)



		# Store critic values
		values_a = values_a.write(t, tf.squeeze(value_a))

		# Store log probability of the action chosen
		action_probs_a = action_probs_a.write(t, action_probs_t_a[0, action_a])

		# Store critic values
		values_b = values_b.write(t, tf.squeeze(value_b))

		# Store log probability of the action chosen
		action_probs_b = action_probs_b.write(t, action_probs_t_b[0, action_b])

		# Apply action to the environment to get next state and reward
		state_a, reward_a, done_a = tf_env_step(action_a,'a')
		# state_a.set_shape(initial_state_shape_a)

		if tf.cast(done_a,tf.bool) :
			First_Bool = True

		py_env.render()
		pygame.display.update()

		state_b, reward_b, done_b = tf_env_step(action_b,'b')
		py_env.render()
		if tf.cast(done_b,tf.bool) :
			Second_Bool = True
		pygame.display.update()
		# state_b.set_shape(initial_state_shape_b)

		# Store reward
		rewards_a = rewards_a.write(t, reward_a)
		rewards_b = rewards_b.write(t, reward_b)
		# print('rewards_a' , reward_a, 'rewards_b' , reward_b)
		if First_Bool and Second_Bool:
			print('break....................................................')
			
			First_Bool = False
			Second_Bool = False
			break
		

	action_probs_a = action_probs_a.stack()
	values_a = values_a.stack()
	rewards_a = rewards_a.stack()

	action_probs_b = action_probs_b.stack()
	values_b = values_b.stack()
	rewards_b = rewards_b.stack()

	return action_probs_a, values_a, rewards_a ,action_probs_b, values_b, rewards_b

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
huber_loss = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)

def compute_loss(action_probs,  values,returns):
	"""Computes the combined actor-critic loss."""

	advantage = returns - values

	action_log_probs = tf.math.log(action_probs)
	actor_loss = -tf.math.reduce_sum(action_log_probs * advantage)

	critic_loss = huber_loss(values, returns)

	return actor_loss + critic_loss

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)


# @tf.function
def train_step(initial_state_a,initial_state_b, modelA , modelB, optimizer, gamma, max_steps_per_episode):
	"""Runs a model training step."""

	with tf.GradientTape(persistent=True) as tape:

		# Run the model for one episode to collect training data
		action_probs_a, values_a, rewards_a ,action_probs_b, values_b, rewards_b = run_episode(initial_state_a , initial_state_b, modelA ,modelB, max_steps_per_episode) 

		# Calculate expected returns
		returns_a = get_expected_return(rewards_a, gamma)
		returns_b = get_expected_return(rewards_b, gamma)

		# Convert training data to appropriate TF tensor shapes
		action_probs_a, values_a, returns_a = [ tf.expand_dims(x, 1) for x in [action_probs_a, values_a, returns_a]] 
		action_probs_b, values_b, returns_b = [ tf.expand_dims(x, 1) for x in [action_probs_b, values_b, returns_b]] 

		# Calculating loss values to update our network
		loss_a = compute_loss(action_probs_a, values_a, returns_a)
		loss_b = compute_loss(action_probs_b, values_b, returns_b)

		# Compute the gradients from the loss
	grads_a = tape.gradient(loss_a, modelA.trainable_variables)
	# print('returns_b' , returns_b , 'loss_b',loss_b) 
	# Apply the gradients to the model's parameters
	optimizer.apply_gradients(zip(grads_a, modelA.trainable_variables))

	grads_b = tape.gradient(loss_b, modelB.trainable_variables)
	# print('grads_a',grads_a,'grads_b',grads_b)
	# Apply the gradients to the model's parameters
	optimizer.apply_gradients(zip(grads_b, modelB.trainable_variables))

	del tape
	# print('rewards_a' , rewards_a)
	episode_reward_a = tf.math.reduce_sum(rewards_a)
	episode_reward_b = tf.math.reduce_sum(rewards_b)

	return episode_reward_a ,episode_reward_b





min_episodes_criterion = 50
max_episodes = 50
max_steps_per_episode = 10000

# Cartpole-v0 is considered solved if average reward is >= 195 over 100 
# consecutive trials
reward_threshold = 195
running_reward = 0

# Discount factor for future rewards
gamma = 0.99

# Keep last episodes reward
episodes_rewardA: collections.deque = collections.deque(maxlen=min_episodes_criterion)
episodes_rewardB: collections.deque = collections.deque(maxlen=min_episodes_criterion)

def test_model(modelA , modelB):
	print('Running Test model ')
	for _ in range(4):
		state_a = tf.constant(tf_env_reset('a'))
		state_b = tf.constant(tf_env_reset('b'))
		done = False
		done_a_a = False
		done_b_b = False
		while not done:
			state_a = tf.expand_dims(state_a, 0)
			state_b = tf.expand_dims(state_b, 0)
			action_logits_t_a,_ = modelA(state_a)
			action_logits_t_b,_ = modelB(state_b) 
			if action_logits_t_a.shape == (1,1,2):
				# action_a = tf.random.categorical(action_logits_t_a[0], 1)[0,0]
				action_a = np.argmax(np.squeeze(action_logits_t_a[0]))
				action_probs_t_a = tf.nn.softmax(action_logits_t_a[0])
				# print('action_probs_t_a' , action_probs_t_a)

				action_b = np.argmax(np.squeeze(action_logits_t_b[0]))
				action_probs_t_b = tf.nn.softmax(action_logits_t_b[0])

			else:
				action_a = np.argmax(np.squeeze(action_logits_t_a))
				action_probs_t_a = tf.nn.softmax(action_logits_t_a)
				# print('action_probs_t_a' , action_probs_t_a)

				action_b = np.argmax(np.squeeze(action_logits_t_b))
				action_probs_t_b = tf.nn.softmax(action_logits_t_b)

			state_a, reward_a, done_a = tf_env_step(action_a,'a')
			if done_a:
				done_a_a = True
			
			py_env.render()
			pygame.display.update()
			if reward_a>1:
				print('good job paddle_a...........................................................................')
				pass

			state_b, reward_b, done_b = tf_env_step(action_b,'b')
			if done_b:
				done_b_b = True
			if reward_b>1:
				print('good job paddle_b...........................................................................')
				pass
			if done_a_a and  done_b_b:
				done = True
			
			





with tqdm.trange(max_episodes) as t:
	for i in t:
		initial_state_a = tf.constant(tf_env_reset('a'))
		initial_state_b = tf.constant(tf_env_reset('b'), dtype=tf.float32)

		episode_reward_a , episode_reward_b = train_step(initial_state_a,initial_state_b, modelA , modelB, optimizer, gamma, max_steps_per_episode)
		episode_reward_a = int(episode_reward_a)
		episode_reward_b = int(episode_reward_b)
		episodes_rewardA.append(episode_reward_a)
		running_reward_a = statistics.mean(episodes_rewardA)
		episodes_rewardB.append(episode_reward_b)
		running_reward_b = statistics.mean(episodes_rewardB)

		t.set_description(f'Episode {i}')
		t.set_postfix(episode_reward=episode_reward_a, running_reward=running_reward_a)
		t.set_postfix(episode_reward=episode_reward_b, running_reward=running_reward_b)

		# Show average episode reward every 10 episodes
		if i % 10 == 0:
			pass # print(f'Episode {i}: average reward: {avg_reward}')
		if i%2==0:
			test_model(modelA , modelB)


		if running_reward_a > reward_threshold and running_reward_a > reward_threshold and i >= min_episodes_criterion:  
			break

print(f'\nSolved at episode {i}: average reward: {running_reward:.2f}!')






















	
