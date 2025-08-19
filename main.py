import random
from typing import List
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

SEED = 42

random.seed(SEED)              # Python random
torch.manual_seed(SEED)        # PyTorch CPU
torch.cuda.manual_seed(SEED)   # PyTorch GPU
torch.cuda.manual_seed_all(SEED)

BASIC_MOVES = ['U', 'U\'', 'D', 'D\'', 'F', 'F\'', 'B', 'B\'', 'L', 'L\'', 'R', 'R\'']
INVERSE_MOVES = {
    'U': 'U\'',
    'U\'' : 'U',
    'D': 'D\'',
    'D\'': 'D',
    'F': 'F\'',
    'F\'': 'F',
    'B': 'B\'',
    'B\'': 'B',
    'L': 'L\'',
    'L\'': 'L',
    'R': 'R\'',
    'R\'': 'R'
}

# Creating a representation of a 2x2 Rubik's Cube
class Cube2x2:
    def __init__(self, stickers = None):
        # If permutation is not given, initialize to solved state
        # Stickers are represented as numbers from 0 to 5
        # each face has 4, top left, top right, bottom left, bottom right
        # order: U, D, F, B, L, R
        self.stickers = stickers or [0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5]

    def is_solved(self):
        bool = True
        for i in range(0,24,4):
            for j in range(i,i+4):
                if self.stickers[j] != self.stickers[i]:
                    bool = False
        return bool


    #Print cube to debug
    def print_cube(cube):

        print("       ", str(cube.stickers[0]).rjust(2), str(cube.stickers[1]).rjust(2))
        print("       ", str(cube.stickers[2]).rjust(2), str(cube.stickers[3]).rjust(2))

        print(str(cube.stickers[16]).rjust(2), str(cube.stickers[17]).rjust(2), " ",
              str(cube.stickers[8]).rjust(2), str(cube.stickers[9]).rjust(2), " ",
              str(cube.stickers[20]).rjust(2), str(cube.stickers[21]).rjust(2), " ",
              str(cube.stickers[12]).rjust(2), str(cube.stickers[13]).rjust(2))

        print(str(cube.stickers[18]).rjust(2), str(cube.stickers[19]).rjust(2), " ",
              str(cube.stickers[10]).rjust(2), str(cube.stickers[11]).rjust(2), " ",
              str(cube.stickers[22]).rjust(2), str(cube.stickers[23]).rjust(2), " ",
              str(cube.stickers[14]).rjust(2), str(cube.stickers[15]).rjust(2))

        print("       ", str(cube.stickers[4]).rjust(2), str(cube.stickers[5]).rjust(2))
        print("       ", str(cube.stickers[6]).rjust(2), str(cube.stickers[7]).rjust(2))
    

def apply_move(cube: Cube2x2, move: str) -> Cube2x2:
    # Define the effect of each move on the cube's stickers
    # turning a 2x2 essentially rotates half of the 3d cube, affecting all faces but the 1 opposite face
    if move == "U":
        cube.stickers = [cube.stickers[2], cube.stickers[0], # u
                         cube.stickers[3], cube.stickers[1],

                         cube.stickers[4], cube.stickers[5], # d
                         cube.stickers[6], cube.stickers[7],

                         cube.stickers[20], cube.stickers[21],  # f
                         cube.stickers[10], cube.stickers[11],

                         cube.stickers[16], cube.stickers[17], # b
                         cube.stickers[14], cube.stickers[15],

                         cube.stickers[8], cube.stickers[9], # l
                         cube.stickers[18], cube.stickers[19],

                         cube.stickers[12], cube.stickers[13], # r
                         cube.stickers[22], cube.stickers[23]]
    elif move == "U'":
        cube.stickers = [cube.stickers[1], cube.stickers[3], # u
                         cube.stickers[0], cube.stickers[2],

                         cube.stickers[4], cube.stickers[5], # d
                         cube.stickers[6], cube.stickers[7],

                         cube.stickers[16], cube.stickers[17],  # f
                         cube.stickers[10], cube.stickers[11],

                         cube.stickers[20], cube.stickers[21], # b
                         cube.stickers[14], cube.stickers[15],

                         cube.stickers[12], cube.stickers[13], # l
                         cube.stickers[18], cube.stickers[19],

                         cube.stickers[8], cube.stickers[9], # r
                         cube.stickers[22], cube.stickers[23]]
    elif move == "D":
        cube.stickers = [cube.stickers[0], cube.stickers[1], # u
                         cube.stickers[2], cube.stickers[3],

                         cube.stickers[6], cube.stickers[4], # d
                         cube.stickers[7], cube.stickers[5],

                         cube.stickers[8], cube.stickers[9],  # f
                         cube.stickers[18], cube.stickers[19],

                         cube.stickers[12], cube.stickers[13], # b
                         cube.stickers[22], cube.stickers[23],

                         cube.stickers[16], cube.stickers[17], # l
                         cube.stickers[14], cube.stickers[15],

                         cube.stickers[20], cube.stickers[21], # r
                         cube.stickers[10], cube.stickers[11]]
    elif move == "D'":
        cube.stickers = [cube.stickers[0], cube.stickers[1], # u
                         cube.stickers[2], cube.stickers[3],

                         cube.stickers[5], cube.stickers[7], # d
                         cube.stickers[4], cube.stickers[6],

                         cube.stickers[8], cube.stickers[9],  # f
                         cube.stickers[22], cube.stickers[23],

                         cube.stickers[12], cube.stickers[13], # b
                         cube.stickers[18], cube.stickers[19],

                         cube.stickers[16], cube.stickers[17], # l
                         cube.stickers[10], cube.stickers[11],

                         cube.stickers[20], cube.stickers[21], # r
                         cube.stickers[14], cube.stickers[15]]
    elif move == "F":
        cube.stickers = [cube.stickers[0], cube.stickers[1], # u
                         cube.stickers[19], cube.stickers[17],

                         cube.stickers[22], cube.stickers[20], # d
                         cube.stickers[6], cube.stickers[7],

                         cube.stickers[10], cube.stickers[8],  # f
                         cube.stickers[11], cube.stickers[9],

                         cube.stickers[12], cube.stickers[13], # b
                         cube.stickers[14], cube.stickers[15],

                         cube.stickers[16], cube.stickers[4], # l
                         cube.stickers[18], cube.stickers[5],

                         cube.stickers[2], cube.stickers[21], # r
                         cube.stickers[3], cube.stickers[23]]
    elif move == "F'":
        cube.stickers = [cube.stickers[0], cube.stickers[1], # u
                         cube.stickers[20], cube.stickers[22],

                         cube.stickers[17], cube.stickers[19], # d
                         cube.stickers[6], cube.stickers[7],

                         cube.stickers[9], cube.stickers[11],  # f
                         cube.stickers[8], cube.stickers[10],

                         cube.stickers[12], cube.stickers[13], # b
                         cube.stickers[14], cube.stickers[15],

                         cube.stickers[16], cube.stickers[3], # l
                         cube.stickers[18], cube.stickers[2],

                         cube.stickers[5], cube.stickers[21], # r
                         cube.stickers[4], cube.stickers[23]]
    elif move == "B":
        cube.stickers = [cube.stickers[21], cube.stickers[23], # u
                         cube.stickers[2], cube.stickers[3],

                         cube.stickers[4], cube.stickers[5], # d
                         cube.stickers[16], cube.stickers[18],

                         cube.stickers[8], cube.stickers[9],  # f
                         cube.stickers[10], cube.stickers[11],

                         cube.stickers[14], cube.stickers[12], # b
                         cube.stickers[15], cube.stickers[13],

                         cube.stickers[1], cube.stickers[17], # l
                         cube.stickers[0], cube.stickers[19],

                         cube.stickers[20], cube.stickers[7], # r
                         cube.stickers[22], cube.stickers[6]]
    elif move == "B'":
        cube.stickers = [cube.stickers[18], cube.stickers[16], # u
                         cube.stickers[2], cube.stickers[3],

                         cube.stickers[4], cube.stickers[5], # d
                         cube.stickers[23], cube.stickers[21],

                         cube.stickers[8], cube.stickers[9],  # f
                         cube.stickers[10], cube.stickers[11],

                         cube.stickers[13], cube.stickers[15], # b
                         cube.stickers[12], cube.stickers[14],

                         cube.stickers[6], cube.stickers[17], # l
                         cube.stickers[7], cube.stickers[19],

                         cube.stickers[20], cube.stickers[0], # r
                         cube.stickers[22], cube.stickers[1]]
    elif move == "L":
        cube.stickers = [cube.stickers[15], cube.stickers[1], # u
                         cube.stickers[13], cube.stickers[3],

                         cube.stickers[8], cube.stickers[5], # d
                         cube.stickers[10], cube.stickers[7],

                         cube.stickers[0], cube.stickers[9],  # f
                         cube.stickers[2], cube.stickers[11],

                         cube.stickers[12], cube.stickers[6], # b
                         cube.stickers[14], cube.stickers[4],

                         cube.stickers[18], cube.stickers[16], # l
                         cube.stickers[19], cube.stickers[17],

                         cube.stickers[20], cube.stickers[21], # r
                         cube.stickers[22], cube.stickers[23]]
    elif move == "L'":
        cube.stickers = [cube.stickers[8], cube.stickers[1], # u
                         cube.stickers[10], cube.stickers[3],

                         cube.stickers[15], cube.stickers[5], # d
                         cube.stickers[13], cube.stickers[7],

                         cube.stickers[4], cube.stickers[9],  # f
                         cube.stickers[6], cube.stickers[11],

                         cube.stickers[12], cube.stickers[2], # b
                         cube.stickers[14], cube.stickers[0],

                         cube.stickers[17], cube.stickers[19], # l
                         cube.stickers[16], cube.stickers[18],

                         cube.stickers[20], cube.stickers[21], # r
                         cube.stickers[22], cube.stickers[23]]
    elif move == "R":
        cube.stickers = [cube.stickers[0], cube.stickers[9], # u
                         cube.stickers[2], cube.stickers[11],

                         cube.stickers[4], cube.stickers[14], # d
                         cube.stickers[6], cube.stickers[12],

                         cube.stickers[8], cube.stickers[5],  # f
                         cube.stickers[10], cube.stickers[7],

                         cube.stickers[3], cube.stickers[13], # b
                         cube.stickers[1], cube.stickers[15],

                         cube.stickers[16], cube.stickers[17], # l
                         cube.stickers[18], cube.stickers[19],

                         cube.stickers[22], cube.stickers[20], # r
                         cube.stickers[23], cube.stickers[21]]
    elif move == "R'":
        cube.stickers = [cube.stickers[0], cube.stickers[14], # u
                         cube.stickers[2], cube.stickers[12],

                         cube.stickers[4], cube.stickers[9], # d
                         cube.stickers[6], cube.stickers[11],

                         cube.stickers[8], cube.stickers[1],  # f
                         cube.stickers[10], cube.stickers[3],

                         cube.stickers[7], cube.stickers[13], # b
                         cube.stickers[5], cube.stickers[15],

                         cube.stickers[16], cube.stickers[17], # l
                         cube.stickers[18], cube.stickers[19],

                         cube.stickers[21], cube.stickers[23], # r
                         cube.stickers[20], cube.stickers[22]]
        

def apply_algorithm(cube: Cube2x2, algorithm: List[str]) -> Cube2x2:
    for move in algorithm:
        apply_move(cube, move)
    return cube

def generate_scramble() -> List[str]:
    scramble = []
    for _ in range(random.randint(10, 15)):
        scramble.append(random.choice(BASIC_MOVES))
    return scramble

def oneHot_encode(cube: Cube2x2) -> torch.Tensor:
    tensor = torch.zeros((24, 6), dtype=torch.float32)
    for i, sticker in enumerate(cube.stickers):
        tensor[i, sticker] = 1.0
    return tensor.flatten()  # Flatten to a 144-dimensional vector

class CubeDataSet(Dataset):
    def __init__(self, n_samples: int):
        self.data = []
        self.labels = []
        self.move_to_index = {m:i for i,m in enumerate(BASIC_MOVES)}
        for _ in range(n_samples):
            cube = Cube2x2()
            scramble = generate_scramble()
            apply_algorithm(cube, scramble)
            for move in reversed(scramble):
                self.data.append(oneHot_encode(cube))
                self.labels.append(self.move_to_index[INVERSE_MOVES[move]])
                apply_move(cube, INVERSE_MOVES[move])
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):  # <-- FIX
        return self.data[idx], self.labels[idx]
    


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(144, 128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(64, 12),

        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    
def train_model(model: nn.Module, dataset: CubeDataSet, epochs: int = 20, batch_size: int = 32, lr=1e-3, n_samples=50000, device=None):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    ds = CubeDataSet(n_samples)
    n_train = int(0.9 * len(ds))
    n_val = len(ds) - n_train
    train_ds, val_ds = torch.utils.data.random_split(ds, [n_train, n_val])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False) 

    model = NeuralNetwork().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for ep in range(1, epochs + 1):
        model.train()
        total = 0; correct = 0; loss_sum = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = torch.as_tensor(yb, dtype=torch.long, device=device)
            opt.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            opt.step()
            loss_sum += loss.item() * xb.size(0)
            pred = logits.argmax(dim=1)
            correct += (pred == yb).sum().item()
            total += xb.size(0)
        train_acc = correct / total
        train_loss = loss_sum / total

        # validation
        model.eval(); vtotal = 0; vcorrect = 0; vloss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = torch.as_tensor(yb, dtype=torch.long, device=device)
                logits = model(xb)
                loss = criterion(logits, yb)
                vloss += loss.item() * xb.size(0)
                pred = logits.argmax(dim=1)
                vcorrect += (pred == yb).sum().item()
                vtotal += xb.size(0)
        val_acc = vcorrect / vtotal
        val_loss = vloss / vtotal
        print(f"Epoch {ep:02d} | train_loss={train_loss:.4f} acc={train_acc:.3f} | val_loss={val_loss:.4f} acc={val_acc:.3f}")

    return model

if __name__ == "__main__":
    model = NeuralNetwork()
    trained_model = train_model(model, CubeDataSet, epochs=20, batch_size=32, n_samples=50)

torch.save(trained_model.state_dict(), "2x2_solver.pth")
