# Flappy Bird AI

This repository contains an AI-based implementation of the classic *Flappy Bird* game. The AI learns to play the game using the **Neuroevolution of Augmenting Topologies (NEAT)** algorithm. The goal is to train the AI to navigate through the obstacles (pipes) with minimal collisions and high scores.

---

## 📌 Features

- **Game Implementation**: Developed using the popular *Pygame* library for 2D graphics and physics.
- **AI Training**: Utilizes the NEAT algorithm for reinforcement learning.
- **Visualization**: Displays AI's real-time learning and decision-making process during gameplay.
- **Customizability**: Easily configurable parameters for training, like population size and fitness functions.

---

## 🚀 Installation

### Prerequisites
- Python 3.8 or above
- `pygame` library
- `neat-python` library

### Setup Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/Aafimalek/flappy_bird_AI.git
   cd flappy_bird_AI
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the game:
   ```bash
   python flappy_bird_ai.py
   ```

---

## 🧠 How It Works

The AI uses the **NEAT algorithm** for evolving neural networks. Here's how it learns to play:
1. **Inputs to the Neural Network**:
   - Bird's position
   - Distance to the next pipe
   - Vertical position of the next pipe
2. **Outputs**:
   - Whether to "jump" or "do nothing."
3. **Fitness Function**:
   - Rewards the bird for surviving longer and passing more pipes.

The NEAT algorithm evolves the neural network over multiple generations, improving the AI's performance over time.

---

## 📂 Project Structure

```
flappy_bird_AI/
├── assets/                  # Game assets like sprites and sounds
├── config-feedforward.txt   # NEAT configuration file
├── flappy_bird_ai.py        # Main script to run the game
├── README.md                # Project documentation
├── requirements.txt         # List of dependencies
```

---

## 🔧 Configuration

You can modify the `config-feedforward.txt` file to tune the NEAT algorithm:
- **Population size**: Number of AI agents per generation.
- **Fitness function**: Adjust how the AI's performance is evaluated.
- **Mutation rate**: Control the variation in neural networks between generations.

---

## 🌟 Future Enhancements

- Add a leaderboard to compare AI scores.
- Train the AI using advanced reinforcement learning techniques like Deep Q-Learning.

---



## 🤝 Contributing

Contributions are welcome! If you have suggestions or improvements:
1. Fork the repository.
2. Create a new branch for your changes.
3. Submit a pull request.

---

## 📬 Contact

For any questions or feedback, feel free to reach out:
- **Author**: Aafi Malek  
- **Email**: [your-email@example.com](mailto:aafimalek2023@gmail.com.com)  

