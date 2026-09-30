You are going to play a 70-round match of Rock, Paper, Scissors against an AI opponent, using a command-line script in this folder.

## Setup

- Working folder: the `sim` folder that contains `play.js`.
- You may read `play.js` to understand how the game and the opponent work.
- Start a new match first:

```
node play.js reset
```

## Playing a round

```
node play.js rock
node play.js paper
node play.js scissors
```

Each command plays one round. The script prints the round result, some information about the opponent, and the running score. You can also run `node play.js status` at any time. The script may print a TensorFlow.js notice on stderr; you can ignore it (add `2>/dev/null` in bash or `2>$null` in PowerShell).

## Rules

1. Choose every move yourself, one round at a time, based on your own reasoning about what you have seen so far. Run exactly one move command per round and read its output before choosing the next move.
2. Do not write or run any code, script or loop that chooses moves for you, and do not run commands that play more than one round at once.
3. Do not read, modify or delete `state.json`, and do not modify `play.js`.
4. Do not look at any other files or folders in this project besides `play.js`.
5. Play exactly 70 rounds, then stop.

Before each move, write one or two sentences explaining why you chose it.

## When the match is done

Write your results to `results/<your-model-name>.json` (for example `results/gemini-2.5-pro.json`) with this structure:

```json
{
  "player": "<your model name and version>",
  "date": "<YYYY-MM-DD>",
  "rounds_played": 70,
  "score": {
    "player_wins": 0,
    "ai_wins": 0,
    "ties": 0,
    "player_win_rate": 0.0,
    "ai_win_rate": 0.0,
    "tie_rate": 0.0
  },
  "strategy_summary": ["<a few sentences describing the strategy you used and how it evolved>"],
  "score_by_segment": [
    { "rounds": "1-7",   "player_wins": 0, "ai_wins": 0, "ties": 0 },
    { "rounds": "8-35",  "player_wins": 0, "ai_wins": 0, "ties": 0 },
    { "rounds": "36-70", "player_wins": 0, "ai_wins": 0, "ties": 0 }
  ],
  "rounds": [
    {
      "round": 1,
      "player": "rock",
      "ai": "paper",
      "winner": "player | ai | tie",
      "ai_expected": "<the move the script said the AI expected, or null>",
      "ai_probs_rock_paper_scissors": [0.0, 0.0, 0.0],
      "rationale": "<why you chose this move>"
    }
  ]
}
```

Fill in `rounds` with all 70 rounds, using the values printed by the script (use `null` for `ai_expected` and `ai_probs_rock_paper_scissors` in rounds where the AI played randomly). Rates are wins divided by 70, rounded to 3 decimals.
