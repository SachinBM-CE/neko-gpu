import torch
import torch.nn as nn

class SACCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 64),  # 2 (state) + 1 (action)
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, state, action):
        # Concatenate state and action
        x = torch.cat([state, action], dim=-1)
        return self.net(x)

# Save the model
model = SACCritic()
scripted = torch.jit.script(model)
scripted.save('critic.pt')