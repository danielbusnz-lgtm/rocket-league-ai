# Step 1: Setup & Understanding RLBot

## What is RLBot?

RLBot is a framework that injects your Python code into Rocket League. It:
1. Reads game data (60 times per second)
2. Gives you: ball position, car positions, boost pads, scores, etc.
3. Takes your controller inputs: throttle, steer, jump, boost, etc.
4. Sends them to the game

## The Game Loop

Every frame (~60 FPS), this happens:

```
1. RLBot reads game state → sends to your bot
2. Your bot decides what to do
3. Your bot returns controller inputs
4. RLBot sends inputs to game
5. Repeat
```

## Understanding Game State

The game gives you data like:

**Ball:**
- Position: (x, y, z) coordinates
- Velocity: how fast it's moving
- Rotation: which way it's spinning

**Your Car:**
- Position, velocity, rotation
- Boost amount (0-100)
- Is on ground? Has jumped?

**Other Cars:**
- Same data for all players

**Environment:**
- Boost pad locations and status
- Time remaining
- Score

## Your First Task: Explore the Data

Before writing ANY code, you need to understand what data you get.

### Installation Check

```bash
# Install RLBot
pip install rlbot

# Launch RLBot GUI
python -m rlbot
```

This opens a window. From here:
1. Click "Start Match"
2. You'll see Rocket League launch
3. Bots will play

### Questions to Think About:

1. **What coordinate system does Rocket League use?**
   - Where is (0, 0, 0)?
   - Which axis is forward/sideways/up?

2. **What inputs can you control?**
   - List all the buttons/axes you can control

3. **How fast is the game?**
   - How many times per second does your code run?

## Next Steps

Once you understand the data flow, we'll build your first bot in `02-basic-bot.md`.

**But first:** Ask me any questions about:
- How RLBot connects to the game
- What data structures you'll work with
- How the coordinate system works
- Anything else that's unclear!

Don't move on until you understand this foundation.
