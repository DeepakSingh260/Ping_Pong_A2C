import pygame 
import numpy as np 
import time 
import tensorflow as tf
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
			reset()

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

while True:
	for event in pygame.event.get():
			if event.type == pygame.QUIT:
				
				pygame.quit()
				quit()
				break

			if event.type == pygame.KEYDOWN:
				if event.key == pygame.K_UP:
					paddle_a.move(-100)
				if event.key == pygame.K_DOWN:
					paddle_a.move(100)
				if event.key == pygame.K_w:
					paddle_b.move(-100)
				if event.key == pygame.K_s:
					paddle_b.move(100)
									
	draw_window()
	ball.move(paddle_a , paddle_b)


class Ping_Pong_Env(py_environment.PyEnvironment):

	def __init__(self):
		self.ball = Ball()
		self.paddle_a = PaddleA()
		self.paddle_b = PaddleB()
		self._action_spec = array_spec.BoundedArraySpec(shape = () , dtype = np.int32 , minimum = 0 ,maximum = 1 , name = 'action')
		self._observation_spec = array_spec.BoundedArraySpec(shape=(3,) , dtype = np.int32 , name = 'observation' )
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
		self.paddle_a = None 
		self.paddle_a = PaddleA()
		self.paddle_b = None
		self.paddle_b = PaddleB()

	def _step(self,action):
		































	