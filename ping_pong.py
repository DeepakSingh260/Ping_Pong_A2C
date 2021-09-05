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

	def _reset(self , s):
		self.ball = None 
		self.ball = Ball()
		self.paddle_a = PaddleA()

		self.paddle_b = PaddleB()

		if s== 'a':
			self._current_time_step = ts.restart(np.array([self.paddle_a.y ,self.paddle_a.x- self.ball.x ,self.paddle_a.y-self.ball.y] ,dtype = np.int32 ))

		if s=='b':
			self._current_time_step = ts.restart(np.array([self.paddle_b.y ,self.paddle_b.x- self.ball.x ,self.paddle_b.y-self.ball.y] ,dtype = np.int32 ))

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

def env_step(action: np.ndarray ,s) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Returns state, reward and done flag given an action."""

	state, reward, done, _ = py_env.step(action,s)
	return (state.astype(np.float32), np.array(reward, np.int32), np.array(done, np.int32))


def tf_env_step(action: tf.Tensor ,s) -> List[tf.Tensor]:

	return tf.numpy_function(env_step, [action , s], [tf.float32, tf.int32, tf.int32])

def run_episode(initial_state,  modelA , modelB, max_steps , s) :

	action_probs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
	values = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
	rewards = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)

	initial_state_shape = initial_state.shape
	state = initial_state

	for t in tf.range(max_steps):
		# Convert state into a batched tensor (batch size = 1)
		state = tf.expand_dims(state, 0)

		# Run the model and to get action probabilities and critic value
		action_logits_t_a, value_a = modelA(state)

		# Sample next action from the action probability distribution
		action_a = tf.random.categorical(action_logits_t_a, 1)[0, 0]
		action_probs_t = tf.nn.softmax(action_logits_t_a)

		# Store critic values
		values = values.write(t, tf.squeeze(value))

		# Store log probability of the action chosen
		action_probs = action_probs.write(t, action_probs_t[0, action])

		# Apply action to the environment to get next state and reward
		state, reward, done = tf_env_step(action,s)
		state.set_shape(initial_state_shape)

		# Store reward
		rewards = rewards.write(t, reward)

		if tf.cast(done, tf.bool):
			break

	action_probs = action_probs.stack()
	values = values.stack()
	rewards = rewards.stack()

	return action_probs, values, rewards
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
def train_step(initial_state, modelA , modelB, optimizer, gamma, max_steps_per_episode):
  """Runs a model training step."""

	with tf.GradientTape() as tape:

		# Run the model for one episode to collect training data
		action_probs, values, rewards = run_episode(initial_state, modelA ,modelB, max_steps_per_episode) 

		# Calculate expected returns
		returns = get_expected_return(rewards, gamma)

		# Convert training data to appropriate TF tensor shapes
		action_probs, values, returns = [ tf.expand_dims(x, 1) for x in [action_probs, values, returns]] 

		# Calculating loss values to update our network
		loss = compute_loss(action_probs, values, returns)

		# Compute the gradients from the loss
	grads = tape.gradient(loss, model.trainable_variables)

	# Apply the gradients to the model's parameters
	optimizer.apply_gradients(zip(grads, model.trainable_variables))

	episode_reward = tf.math.reduce_sum(rewards)

	return episode_reward





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
episodes_reward: collections.deque = collections.deque(maxlen=min_episodes_criterion)

with tqdm.trange(max_episodes) as t:
	for i in t:
	initial_state = tf.constant(env.reset(), dtype=tf.float32)
	episode_reward = int(train_step(
	    initial_state, modelA , modelB, optimizer, gamma, max_steps_per_episode))

	episodes_reward.append(episode_reward)
	running_reward = statistics.mean(episodes_reward)

	t.set_description(f'Episode {i}')
	t.set_postfix(episode_reward=episode_reward, running_reward=running_reward)

	# Show average episode reward every 10 episodes
	if i % 10 == 0:
		pass # print(f'Episode {i}: average reward: {avg_reward}')

	if running_reward > reward_threshold and i >= min_episodes_criterion:  
		break

print(f'\nSolved at episode {i}: average reward: {running_reward:.2f}!')






















	