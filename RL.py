#Hyperparameters
learning_rate = 0.01
gamma = 0.99
class Policy(nn.Module):
	def __init__(self):
		super(Policy, self).__init__()
		self.state_space = env.observation_space.shape[0]
		self.action_space = env.action_space.n
		
		self.l1 = nn.Linear(self.state_space, 128, bias=False)
		self.l2 = nn.Linear(128, self.action_space, bias=False)
		
		self.gamma = gamma
		
		# Episode policy and reward history 
		self.policy_history = Variable(torch.Tensor()) 
		self.reward_episode = []
		# Overall reward and loss history
		self.reward_history = []
		self.loss_history = []
def forward(self, x):	
		model = torch.nn.Sequential(
			self.l1,
			nn.Dropout(p=0.6),
			nn.ReLU(),
			self.l2,
			nn.Softmax(dim=-1)
		)
		return model(x)
policy = Policy()
optimizer = optim.Adam(policy.parameters(), lr=learning_rate)
