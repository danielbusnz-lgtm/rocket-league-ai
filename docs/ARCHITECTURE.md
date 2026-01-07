# Rocket League Champion-Level Bot Architecture

**Target Level:** Champion (Top 5% of players)
**Hardware:** RTX 5080
**Timeline:** 4-6 months
**Training Steps:** 15-20M steps

---

## Architecture Choice: Transformer Actor-Critic

### Why Transformer Over LSTM

**Advantages:**
- **Selective attention:** Learns what to focus on dynamically
- **Long-range dependencies:** Direct connections to any past frame
- **Parallelization:** Faster training on GPU
- **Interpretability:** Can visualize attention weights

**Tradeoffs:**
- Needs more data (~15-20M steps vs 10M for LSTM)
- More hyperparameters to tune
- Larger memory footprint
- Higher ceiling but slower initial learning

---

## Full Network Architecture

### Input Processing
```
Sequence: 16 frames (267ms at 60fps)
Per frame: 80 features
Input shape: (batch, 16, 80)
```

### Embedding Layer
```
Linear projection: 80 → 256 per frame
+ Learnable positional encoding (16, 256)
Output shape: (batch, 16, 256)
```

### Transformer Encoder

**Layer 1:**
```
Multi-Head Self-Attention:
  - 4 heads × 64 dimensions
  - Causal masking (no future peeking)
  - Pre-norm LayerNorm

Feedforward Network:
  - Linear(256 → 1024) + GELU
  - Linear(1024 → 256)
  - Pre-norm LayerNorm
  - Dropout(0.1) during training
```

**Layer 2:** (Same structure as Layer 1)

**Output:** (batch, 16, 256)

### Pooling
```
Take last frame embedding: output[:, -1, :]
Shape: (batch, 256)
```

### Actor Head (Policy Network)

**Shared layers:**
```
Dense(256 → 256, ReLU)
Dense(256 → 128, ReLU)
```

**Continuous actions (throttle, steer, pitch, yaw, roll):**
```
Output: 10 values (5 means + 5 log_stds)
Mean: tanh(output[:5]) → [-1, 1]
Std: exp(output[5:10]) → (0, ∞)
Distribution: Normal(mean, std)
```

**Discrete actions (jump, boost, handbrake):**
```
Output: 3 values
Probability: sigmoid(output) → [0, 1]
Distribution: Bernoulli(p)
```

### Critic Head (Value Network)
```
Dense(256 → 256, ReLU)
Dense(256 → 128, ReLU)
Dense(128 → 1, Linear)
Output: State value V(s)
```

---

## Input Features (80 Total)

### 1. Ball Features (12)
- Position (x, y, z) - normalized to [-1, 1]
- Velocity (vx, vy, vz) - normalized to [-1, 1]
- Angular velocity (wx, wy, wz) - normalized
- Distance from origin - normalized to [0, 1]
- Speed (scalar) - normalized to [0, 1]
- Is moving fast (>1000) - binary

**Normalization:**
- Position X: (x + 4096) / 8192 * 2 - 1
- Position Y: (y + 5120) / 10240 * 2 - 1
- Position Z: z / 2044
- Velocity: v / 2300, clipped to [-1, 1]

### 2. Bot Features (15)
- Position (x, y, z) - normalized
- Velocity (vx, vy, vz) - normalized
- Forward vector (3 components) - from rotation
- Up vector (3 components) - from rotation
- Angular velocity (wx, wy, wz) - normalized
- Boost amount - [0, 100] → [0, 1]
- On ground - binary
- Has jumped - binary
- Has double jumped - binary

### 3. Spatial Relationship Features (10)
- Distance to ball - normalized
- Horizontal angle to ball - sin/cos encoding (2 values)
- Vertical angle to ball - sin/cos encoding (2 values)
- Velocity toward ball - dot product, normalized
- Ball approaching bot - binary
- Time to ball (estimated) - normalized
- Ball height category - one-hot or normalized

### 4. Ball Trajectory Prediction (15)
- Ball position at t+0.5s (x, y, z) - normalized
- Ball position at t+1.0s (x, y, z) - normalized
- Ball position at t+1.5s (x, y, z) - normalized
- Ball velocity at t+1.0s (vx, vy, vz) - normalized
- Will bounce soon - binary
- Trajectory toward goal - binary

### 5. Game State Features (8)
- Time elapsed - [0, 300] → [0, 1]
- Time remaining - normalized
- Score difference - normalized to [-1, 1]
- Is kickoff - binary
- Is round active - binary
- Am I closest to ball - binary
- Am I furthest from ball - binary
- Last touch was mine - binary

### 6. Boost Pad Features (20)
For 5 nearest boost pads (4 features each):
- Distance to pad - normalized
- Is active - binary
- Is big pad - binary
- Angle to pad - sin/cos encoding (counted as 1 logical feature)

---

## Multi-Head Self-Attention Details

### The Math
```
For each head h ∈ {1,2,3,4}:
  Q = X @ W_Q^h  (256 → 64)  # Query
  K = X @ W_K^h  (256 → 64)  # Key
  V = X @ W_V^h  (256 → 64)  # Value

  scores = (Q @ K^T) / sqrt(64)  # (batch, 16, 16)
  scores = scores.masked_fill(causal_mask == 0, -inf)
  weights = softmax(scores, dim=-1)
  output^h = weights @ V  # (batch, 16, 64)

Concatenate heads:
  output = concat(output^1, ..., output^4)  # (batch, 16, 256)
  output = output @ W_O  # (256 → 256)
```

### Causal Masking
```
Mask prevents attention to future frames:

[[1, 0, 0, 0, ...]   # Frame 0 can only see itself
 [1, 1, 0, 0, ...]   # Frame 1 can see 0, 1
 [1, 1, 1, 0, ...]   # Frame 2 can see 0, 1, 2
 [1, 1, 1, 1, ...]]  # etc.
```

### Positional Encoding
```
Learnable embeddings: nn.Parameter(torch.randn(16, 256))
Added to frame embeddings: embedded = frame_emb + pos_emb[t]
```

---

## PPO Training Algorithm

### Hyperparameters
```
Learning rate: 3e-4
  - Cosine annealing with warmup
  - Warmup steps: 10,000
  - Decay to 1e-5 over 10M steps

Optimizer: AdamW
  - β1: 0.9
  - β2: 0.999
  - Weight decay: 1e-4

Batch size: 4096 steps
Minibatch size: 128
PPO epochs: 4
PPO clip ε: 0.2

GAE lambda: 0.95
Discount γ: 0.99

Gradient clipping: 0.5 max norm

Entropy coefficient: 0.01 → 0.001 (decay over training)
Value loss coefficient: 0.5
```

### Training Loop
```
For each iteration:
  1. Collect 4096 steps across 16 parallel environments
  2. Compute returns and advantages using GAE
  3. For 4 epochs:
     - Shuffle data
     - For each minibatch of 128:
       - Forward pass
       - Compute losses (policy + value + entropy)
       - Backprop with gradient clipping
       - Update weights
  4. Update learning rate (cosine schedule)
  5. Log metrics
```

### Loss Functions

**Policy Loss (Clipped PPO):**
```
ratio = π_new(a|s) / π_old(a|s)
clipped_ratio = clip(ratio, 1-ε, 1+ε)
L_policy = -min(ratio * A, clipped_ratio * A)
```

**Value Loss:**
```
L_value = (V(s) - V_target)²
```

**Entropy Loss (for exploration):**
```
L_entropy = -H(π)
```

**Total Loss:**
```
L = L_policy + 0.5 * L_value - c_entropy * L_entropy
```

### GAE (Generalized Advantage Estimation)
```
δ_t = r_t + γ V(s_{t+1}) - V(s_t)
A_t = Σ_{l=0}^∞ (γλ)^l δ_{t+l}
```

---

## Reward Function Design

### Primary Rewards (Sparse)
```
Goal scored:          +100
Goal conceded:        -100
Shot on goal:         +10
Save:                 +20
```

### Mechanical Rewards (Medium Frequency)
```
Ball touch toward goal:     +5
Ball touch toward own goal: -3
Aerial hit:                 +3
Fast aerial:                +5
Demo opponent:              +5
```

### Positioning Rewards (Dense)
```
Good defensive position:  +0.1 per frame
Good offensive position:  +0.1 per frame
Too far from play:        -0.05 per frame
```

### Boost Management
```
Collect small boost pad:  +0.5
Collect big boost pad:    +1.0
Waste boost:              -0.01 per frame
```

### Velocity/Movement Rewards
```
High velocity toward ball:            +0.02 per frame
High velocity toward good position:   +0.01 per frame
Moving toward own goal unnecessarily: -0.02 per frame
```

### Advanced Mechanics (Added Later)
```
Half flip:        +2
Wave dash:        +1.5
Wall hit:         +3
Ceiling shot:     +8
Flip reset:       +10
Air dribble:      +7
```

### Reward Tuning Principles
1. Sparse + Dense: Big rewards for goals, small rewards guide behavior
2. Magnitude hierarchy: Goals > Touches > Positioning
3. No contradictions: Don't reward opposing behaviors
4. Curriculum compatible: Start simple, add complexity

---

## Curriculum Learning Strategy

### Stage 1: Ball Control (0-2M steps)
**Environment:**
- No opponents
- Ball spawns randomly
- Episode: 30 seconds

**Rewards:** Ball touch, distance to ball, velocity toward ball

**Success:** 90% ball touch rate, avg distance < 1000 units

### Stage 2: Directed Hits (2M-4M steps)
**Environment:**
- Empty goal
- Ball spawns various positions
- Episode: 60 seconds

**Rewards:** Add directional rewards, shots on goal

**Success:** 50% shot accuracy, aerials up to 500 units

### Stage 3: Aerials (4M-6M steps)
**Environment:**
- Ball spawns high (500-1500 units)
- Must aerial to hit
- Episode: 45 seconds

**Rewards:** High aerial rewards, fast aerial bonus

**Success:** 60% aerial hit rate, fast aerials

### Stage 4: Defense (6M-8M steps)
**Environment:**
- Bot defends goal
- Opponent bot shoots
- Episode: 90 seconds

**Rewards:** Saves, defensive positioning, clears

**Success:** 40% save rate vs Pro bots

### Stage 5: Self-Play (8M-15M steps)
**Environment:**
- Bot vs itself (previous versions)
- Full 1v1 games
- Episode: 5 minutes

**Rewards:** All rewards active

**Success:** Winning vs older versions, >3 goals per game

### Stage 6: Polish (15M-20M steps)
**Environment:**
- Mix of self-play and fixed opponents
- Longer episodes
- Fine-tune rewards

**Success:** Consistent Champion-level mechanics and game sense

---

## Training Infrastructure

### Parallel Environments
```
Setup:
- 16 parallel Rocket League instances
- Each at 120 FPS (2x speed)
- Central trainer on GPU
- Workers on CPU cores

Communication:
- Workers → Trainer: (state_seq, action, reward, done)
- Trainer → Workers: Updated network weights
- Use multiprocessing Queue or shared memory
```

### Training Speed Estimates
```
16 envs × 120 FPS × 60s = 115,200 steps/minute
Target: 20M steps
Time: ~3 hours theoretical

Realistic (with overhead):
- Network updates slow this down
- Environment resets add overhead
- Expect 10-14 days continuous training
```

### Checkpointing Strategy
```
Save every 100K steps:
- Network weights
- Optimizer state
- Training metrics
- Replay buffer (optional)

Evaluate every 500K steps:
- Play vs Allstar bots
- Record win rate and mechanics
- Keep best performing checkpoint
```

---

## Memory Requirements (RTX 5080)

### Model Size
```
Total parameters: ~2.5M
Memory: ~10MB (fp32 weights)
```

### Training Batch Memory
```
Input: 4096 × 16 × 80 = 21MB
Embeddings: 4096 × 16 × 256 = 67MB
Attention: ~200MB
FFN: ~150MB
Gradients: ~10MB
Optimizer state: ~20MB

Peak: ~500MB

Your 5080 (16GB) can easily handle:
- Batch size up to 8192
- Larger model (512 dims)
- 32 parallel environments
```

---

## Implementation Roadmap

### Phase 1: Foundation (Week 1-2)
1. State processor (80 features from GameTickPacket)
2. Transformer network in PyTorch
3. Forward pass testing
4. RLBot integration

### Phase 2: Training Infrastructure (Week 3-4)
1. PPO algorithm implementation
2. Parallel environment wrapper
3. Experience collection
4. GAE calculation
5. Training loop

### Phase 3: Initial Training (Week 5-8)
1. Stage 1: Ball control
2. Reward function tuning
3. Debug NaNs and instabilities
4. Hyperparameter adjustment

### Phase 4: Curriculum (Week 9-16)
1. Progress through stages 2-5
2. Continuous reward tuning
3. Monitor metrics
4. Iterate on architecture if needed

### Phase 5: Polish (Week 17-24)
1. Stage 6: Self-play and fine-tuning
2. Test against various opponents
3. Optimize performance
4. Reach Champion level

---

## Key Technical Challenges

### 1. Action Space (Continuous + Discrete)
**Solution:**
- Separate output heads
- Normal distribution for continuous
- Bernoulli for discrete
- Combined log probability for PPO

### 2. LSTM Hidden State Management
Not applicable (using Transformer)
- Sequence-based, stateless between episodes
- Easier than LSTM state management

### 3. Trajectory Prediction
**Solution:**
- Use RLBot's ball prediction API
- Or implement simple physics simulator
- Cache predictions for efficiency

### 4. Exploration vs Exploitation
**Solution:**
- Entropy regularization (decay from 0.01 to 0.001)
- Action noise during training
- PPO naturally balances via KL constraint

### 5. Reward Shaping
**Solution:**
- Start simple, iterate constantly
- Log reward magnitudes
- Watch training, identify exploits
- Adjust based on behavior

---

## Success Metrics

### Performance Metrics
- Episode reward (trend: increasing)
- Goals scored per episode (target: >3)
- Goals conceded per episode (target: <2)
- Shot accuracy (target: >50%)
- Save percentage (target: >40%)

### Behavior Metrics
- Average distance to ball
- Aerial hit rate (target: >60%)
- Boost collection rate
- Boost efficiency
- Time spent supersonic

### Network Metrics
- Policy loss (should decrease then stabilize)
- Value loss (should decrease)
- Entropy (gradual decrease)
- KL divergence (should stay < 0.02)
- Gradient norms (check for explosions)

---

## File Structure

```
rocket-league-ai/
├── config/
│   ├── bot.cfg                 # RLBot configuration
│   └── training.yaml           # Training hyperparameters
├── src/
│   ├── state_processor.py      # 80 feature extraction
│   ├── network.py              # Transformer Actor-Critic
│   ├── ppo.py                  # PPO training algorithm
│   ├── environment.py          # RLBot environment wrapper
│   ├── trainer.py              # Main training loop
│   ├── rewards.py              # Reward function
│   ├── curriculum.py           # Stage management
│   └── bot.py                  # RLBot agent
├── docs/
│   ├── ARCHITECTURE.md         # This file
│   ├── 01-setup.md            # Setup guide
│   └── training-log.md        # Training progress
├── models/                     # Saved checkpoints
├── logs/                       # TensorBoard logs
└── tests/                      # Unit tests
```

---

## Next Steps

1. ✅ Architecture defined
2. ⏳ Implement state_processor.py
3. ⏳ Implement network.py
4. ⏳ Implement ppo.py
5. ⏳ Implement training infrastructure
6. ⏳ Begin training Stage 1
7. ⏳ Iterate and reach Champion level

---

## References

- **Original Transformer:** "Attention is All You Need" (Vaswani et al., 2017)
- **PPO Algorithm:** "Proximal Policy Optimization" (Schulman et al., 2017)
- **GAE:** "High-Dimensional Continuous Control Using Generalized Advantage Estimation" (Schulman et al., 2016)
- **RLBot Documentation:** https://rlbot.org/
- **Top RL Bots:** Nexto, Necto (for reference and inspiration)
