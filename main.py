import random
from typing import List
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from datetime import datetime
from copy import copy
import torch.nn.functional as F

length = 0


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

def generate_scramble(scrambleLength = None) -> List[str]:
    scramble = []
    length = random.randint(5,11) if scrambleLength is None else scrambleLength
    for _ in range(length):
        scramble.append(random.choice(BASIC_MOVES))
    return scramble

def oneHot_encode(cube: Cube2x2) -> torch.Tensor:
    # Create (6, 2, 2) one-hot tensor for CNN input
    tensor = torch.zeros((6, 2, 2), dtype=torch.float32)
    for i, sticker in enumerate(cube.stickers):
        face = i // 4              # 0..5
        pos = i % 4                # 0..3
        row = pos // 2              # 0 or 1
        col = pos % 2               # 0 or 1
        tensor[face, row, col] = sticker / 5.0  # normalize 0..5 to 0..1
    return tensor

def copy(self):
    return Cube2x2(self.stickers.copy())

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
                self.data.append(oneHot_encode(cube).view(6,2,2))  # <-- keep as 6x2x2
                self.labels.append(self.move_to_index[INVERSE_MOVES[move]])
                apply_move(cube, INVERSE_MOVES[move])
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], torch.tensor(self.labels[idx], dtype=torch.long)
    


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x):
        identity = x
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        out += identity   # residual connection
        out = F.relu(out)
        return out

# CNN for 2x2 Cube
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # Input: 6 channels (faces), 2x2 stickers per face
        self.conv1 = nn.Conv2d(6, 32, kernel_size=2)  # output: (batch, 32, 1, 1)
        self.resblock1 = ResidualBlock(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=1)  # combine features across channels
        self.resblock2 = ResidualBlock(64)
        
        self.fc1 = nn.Linear(64, 64)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, 12)  # 12 possible moves

    def forward(self, x):
        # x: (batch, 24*6) flattened one-hot vector
        x = x.view(-1, 6, 2, 2)       # reshape to 6 channels, 2x2
        x = F.relu(self.conv1(x))
        x = self.resblock1(x)
        x = F.relu(self.conv2(x))
        x = self.resblock2(x)
        x = x.view(x.size(0), -1)     # flatten
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        logits = self.fc2(x)
        return logits
    
def train_model(model: nn.Module, dataset: CubeDataSet, epochs: int = 20, batch_size: int = 32, lr=1e-3, n_samples=50000, device=None):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)  # added small weight decay
    criterion = nn.CrossEntropyLoss()

    for ep in range(1, epochs + 1):
        ds = CubeDataSet(n_samples)
        n_train = int(0.9 * len(ds))
        n_val = len(ds) - n_train
        train_ds, val_ds = torch.utils.data.random_split(ds, [n_train, n_val])

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
        
        model.train()
        total = 0; correct = 0; loss_sum = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)              # now xb is (batch, 144)
            yb = yb.to(device, dtype=torch.long)
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
        print(f"Epoch {ep:02d} | train_loss={train_loss:.4f} acc={train_acc:.3f} | val_loss={val_loss:.4f} acc={val_acc:.3f} time={datetime.now().time()}")

    return model

def trainTheModel(epochs, n_samples, batch_size):
    
    from pathlib import Path

    MODEL_PATH = Path("models")
    MODEL_PATH.mkdir(parents=True, exist_ok=True)

    MODEL_NAME = "2x2_solver_cnn.pth"
    MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

    # Initialize and train the model
    model = NeuralNetwork()
    model.load_state_dict(torch.load(f=MODEL_SAVE_PATH))  # Load existing model if available
    trained_model = train_model(model, CubeDataSet, epochs=epochs, batch_size=batch_size, n_samples=n_samples)

    torch.save(obj=model.state_dict(), f=MODEL_SAVE_PATH)

def bestMove(cube: Cube2x2) -> str:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = NeuralNetwork().to(device)
    model.load_state_dict(torch.load("models/2x2_solver_cnn.pth", map_location=device))
    model.eval()
    state_tensor = oneHot_encode(cube).unsqueeze(0).to(device)
    move_idx = model(state_tensor).argmax(dim=1).item()
    move = BASIC_MOVES[move_idx]
    return move


def solveCube(cube: Cube2x2, max_moves=100) -> List[str]:
    moves = []
    cube_clone = copy(cube)
    seen_states = set()
    
    while not cube_clone.is_solved() and len(moves) < max_moves:
        state_hash = tuple(cube_clone.stickers)
        if state_hash in seen_states or (moves and bestMove(cube_clone) == INVERSE_MOVES[moves[-1]]):
            move = random.choice(BASIC_MOVES)
        else:
            move = bestMove(cube_clone)
        seen_states.add(state_hash)
        moves.append(move)
        print(moves)
        apply_move(cube_clone, move)
    return moves

def demoSolve(cube, scrambleLength = None):
    print("scramble length =", scrambleLength)
    apply_algorithm(cube, generate_scramble(scrambleLength = scrambleLength))
    Cube2x2.print_cube(cube)
    print("\nSolving the cube...\n")
    solve = solveCube(cube)
    print(solve)
    apply_algorithm(cube, solve)
    print("\n")
    Cube2x2.print_cube(cube)



def bareSolve(cube, scrambleLength=None):
    global length
    apply_algorithm(cube, generate_scramble(scrambleLength=scrambleLength))
    solve = solveCube(cube)
    apply_algorithm(cube, solve)
    length = len(solve)
    return cube.is_solved()

def solveSuccessRate(attempts, scrambleLength=10):
    successes = 0
    totLength = 0
    for i in range(attempts):
        cube = Cube2x2()
        isSolved = bareSolve(cube, scrambleLength=scrambleLength)
        if isSolved:
            successes += 1
            totLength += length
    avg_length = totLength / successes if successes > 0 else 0
    print(f"Success rate: {successes}/{attempts} = {successes/attempts:.2%}",
          f"Average length: {avg_length:.2f} moves per solve")

solveSuccessRate(100, scrambleLength=7)

