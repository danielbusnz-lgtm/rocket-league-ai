# Rocket League AI Bot - Learning Project

## What You're Building

An AI bot that learns to play Rocket League using **Reinforcement Learning**. The bot will:
- Read game state (where's the ball? where am I?)
- Make decisions using a neural network
- Learn from experience (trial and error)

## Prerequisites You Need

1. **Rocket League** (Steam version, Epic version works too)
2. **Python 3.9+** installed
3. **Basic Python knowledge** (functions, classes, loops)
4. **Willingness to learn** about neural networks and RL

## Setup Steps

### 1. Install Dependencies
```bash
cd rocket-league-ai
pip install -r requirements.txt
```

### 2. Install RLBot
The RLBot framework creates a bridge between your Python code and Rocket League:
```bash
pip install rlbot
```

### 3. Verify Installation
```bash
python -c "import rlbot; print('RLBot installed!')"
```

## Learning Path

I've created teaching documents in the `docs/` folder. Follow them in order:

1. **01-setup.md** - Get everything working
2. **02-basic-bot.md** - Build a simple rule-based bot (no AI yet)
3. **03-neural-network.md** - Add a brain to your bot
4. **04-training.md** - Teach your bot through reinforcement learning

## Project Structure

- `config/` - Bot settings and training parameters
- `src/bot/` - Main bot code (YOU write this)
- `src/brain/` - Neural network decision-making (YOU build this)
- `src/training/` - Learning system (YOU implement this)
- `docs/` - Step-by-step teaching materials

## Running Your Bot

```bash
# Start RLBot GUI
python -m rlbot

# Or run directly
python src/bot/bot.py
```

## Questions?

As you build each component, ask me:
- "How do I implement X?"
- "Why does Y work this way?"
- "What's the best approach for Z?"

I'll teach you HOW to write the code, not just give you the solution.

Ready to start? Open `docs/01-setup.md`!
