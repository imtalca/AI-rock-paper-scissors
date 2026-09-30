// Compare the LSTM and the attention model on the SAME fixed move sequences,
// so the result measures the models and not whoever is playing them.
//
//   node compare.js                       150 rounds, 3 seeds
//   node compare.js --rounds 200 --seeds 5
//
// Each run works exactly like the game: the AI commits to its move BEFORE
// seeing the player's move, then the move is added to history and the model
// trains on the whole history (20 epochs). The scripted players ignore the AI.
// The player sequences are seeded; the models' weight initialisation and
// training shuffle are not, so repeated runs differ slightly.
//
// Prints a table and writes results/compare-lstm-vs-attention.json.

const fs = require("fs");
const path = require("path");
const tf = require("@tensorflow/tfjs");

// ---- Copied from index.html ------------------------------------------------

const MOVES = ["rock", "paper", "scissors"];
const WINDOW = 6;
const COUNTER = { rock: "paper", paper: "scissors", scissors: "rock" };

function decideWinner(you, ai) {
  if (you === ai) return "tie";
  const youWin =
    (you === "rock" && ai === "scissors") ||
    (you === "scissors" && ai === "paper") ||
    (you === "paper" && ai === "rock");
  return youWin ? "you" : "ai";
}

function oneHot(move) {
  const vec = [0, 0, 0];
  vec[MOVES.indexOf(move)] = 1;
  return vec;
}

function oneHotWithPosition(move, position) {
  const pos = new Array(WINDOW).fill(0);
  pos[position] = 1;
  return [...oneHot(move), ...pos];
}

function createModel() {
  const model = tf.sequential({
    layers: [
      tf.layers.lstm({ units: 16, inputShape: [WINDOW, 3] }),
      tf.layers.dense({ units: 3, activation: "softmax" }),
    ],
  });
  model.compile({ optimizer: "adam", loss: "categoricalCrossentropy" });
  return model;
}

function createAttentionModel() {
  const x = tf.input({ shape: [WINDOW, 3 + WINDOW] });
  const emb = tf.layers.dense({ units: 8 }).apply(x);
  const q = tf.layers.dense({ units: 8 }).apply(emb);
  const k = tf.layers.dense({ units: 8 }).apply(emb);
  const v = tf.layers.dense({ units: 8 }).apply(emb);
  const scores = tf.layers.dot({ axes: [2, 2] }).apply([q, k]);
  const weights = tf.layers.softmax({ axis: -1 }).apply(scores);
  const attn = tf.layers.dot({ axes: [2, 1] }).apply([weights, v]);
  const flat = tf.layers.flatten().apply(attn);
  const out = tf.layers.dense({ units: 3, activation: "softmax" }).apply(flat);
  const attentionModel = tf.model({ inputs: x, outputs: out });
  attentionModel.compile({ optimizer: "adam", loss: "categoricalCrossentropy" });
  return attentionModel;
}

// The two models differ only in how a window is encoded and how the net is built.
const MODELS = {
  lstm: { create: createModel, encode: moves => moves.map(oneHot) },
  attention: { create: createAttentionModel, encode: moves => moves.map((m, pos) => oneHotWithPosition(m, pos)) },
};

// ---- Scripted players --------------------------------------------------------

// Small seeded random generator, so every model sees the same "random" player.
function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// Each player returns the full move sequence for one match.
const PLAYERS = {
  // rock, paper, scissors, rock, ... : fully predictable
  cycle: (rounds) => Array.from({ length: rounds }, (_, i) => MOVES[i % 3]),

  // 60% rock, 20% paper, 20% scissors: only the bias is predictable (best possible: 60%)
  biased: (rounds, seed) => {
    const rand = mulberry32(seed);
    return Array.from({ length: rounds }, () => {
      const r = rand();
      return r < 0.6 ? "rock" : r < 0.8 ? "paper" : "scissors";
    });
  },

  // cycle for the first half, then the reverse cycle: how fast does the model re-learn?
  switching: (rounds) => Array.from({ length: rounds }, (_, i) =>
    i < rounds / 2 ? MOVES[i % 3] : MOVES[(3 - (i % 3)) % 3]),

  // uniform random: control, nothing to learn (best possible: 33%)
  random: (rounds, seed) => {
    const rand = mulberry32(seed + 1000);
    return Array.from({ length: rounds }, () => MOVES[Math.floor(rand() * 3)]);
  },
};

// ---- One match ---------------------------------------------------------------

async function runMatch(modelName, sequence) {
  const { create, encode } = MODELS[modelName];
  let model = null;
  const history = [];
  const half = sequence.length / 2;
  const stats = { predicted: 0, correct: 0, aiWins: 0, playerWins: 0, ties: 0,
                  firstHalf: { predicted: 0, correct: 0 }, secondHalf: { predicted: 0, correct: 0 } };

  for (let i = 0; i < sequence.length; i++) {
    const yourMove = sequence[i];

    // AI commits first.
    let aiMove;
    if (history.length < WINDOW || model === null) {
      aiMove = MOVES[Math.floor(Math.random() * 3)];
    } else {
      const probs = tf.tidy(() => model.predict(tf.tensor3d([encode(history.slice(-WINDOW))])).dataSync());
      let best = 0;
      for (let j = 1; j < 3; j++) if (probs[j] > probs[best]) best = j;
      const predicted = MOVES[best];
      aiMove = COUNTER[predicted];

      const part = i < half ? stats.firstHalf : stats.secondHalf;
      stats.predicted++; part.predicted++;
      if (predicted === yourMove) { stats.correct++; part.correct++; }
    }

    const winner = decideWinner(yourMove, aiMove);
    if (winner === "ai") stats.aiWins++;
    else if (winner === "you") stats.playerWins++;
    else stats.ties++;

    // Record the move and train on the whole history, like trainModel() in the page.
    history.push(yourMove);
    if (history.length > WINDOW) {
      if (model === null) model = create();
      const xs = [], ys = [];
      for (let j = WINDOW; j < history.length; j++) {
        xs.push(encode(history.slice(j - WINDOW, j)));
        ys.push(oneHot(history[j]));
      }
      const xT = tf.tensor3d(xs), yT = tf.tensor2d(ys);
      await model.fit(xT, yT, { epochs: 20, shuffle: true, verbose: 0 });
      xT.dispose(); yT.dispose();
    }
  }

  if (model) model.dispose();
  return stats;
}

// ---- Main ----------------------------------------------------------------------

function argValue(name, fallback) {
  const i = process.argv.indexOf("--" + name);
  return i === -1 ? fallback : Number(process.argv[i + 1]);
}

const pct = (a, b) => (b === 0 ? 0 : Math.round((a / b) * 1000) / 10);

async function main() {
  await tf.setBackend("cpu");
  const rounds = argValue("rounds", 150);
  const seeds = argValue("seeds", 3);
  console.log(`Rounds per match: ${rounds}   Seeds: ${seeds}\n`);

  const results = [];
  for (const playerName of Object.keys(PLAYERS)) {
    for (const modelName of Object.keys(MODELS)) {
      const total = { predicted: 0, correct: 0, aiWins: 0, playerWins: 0, ties: 0,
                      firstHalf: { predicted: 0, correct: 0 }, secondHalf: { predicted: 0, correct: 0 } };
      const perSeed = [];
      for (let seed = 1; seed <= seeds; seed++) {
        const s = await runMatch(modelName, PLAYERS[playerName](rounds, seed));
        perSeed.push(pct(s.correct, s.predicted));
        for (const key of ["predicted", "correct", "aiWins", "playerWins", "ties"]) total[key] += s[key];
        for (const part of ["firstHalf", "secondHalf"]) {
          total[part].predicted += s[part].predicted;
          total[part].correct += s[part].correct;
        }
      }
      const games = rounds * seeds;
      const row = {
        player: playerName,
        model: modelName,
        prediction_accuracy_pct: pct(total.correct, total.predicted),
        accuracy_first_half_pct: pct(total.firstHalf.correct, total.firstHalf.predicted),
        accuracy_second_half_pct: pct(total.secondHalf.correct, total.secondHalf.predicted),
        accuracy_per_seed_pct: perSeed,
        ai_win_pct: pct(total.aiWins, games),
        tie_pct: pct(total.ties, games),
        player_win_pct: pct(total.playerWins, games),
      };
      results.push(row);
      console.log(
        `${playerName.padEnd(10)} ${modelName.padEnd(10)} ` +
        `accuracy ${String(row.prediction_accuracy_pct).padStart(5)}%  ` +
        `(1st half ${String(row.accuracy_first_half_pct).padStart(5)}%, 2nd half ${String(row.accuracy_second_half_pct).padStart(5)}%)  ` +
        `AI wins ${String(row.ai_win_pct).padStart(5)}%  ties ${String(row.tie_pct).padStart(5)}%  player wins ${String(row.player_win_pct).padStart(5)}%`
      );
    }
  }

  const outFile = path.join(__dirname, "results", "compare-lstm-vs-attention.json");
  fs.writeFileSync(outFile, JSON.stringify({
    description: "LSTM vs attention model on identical scripted move sequences. Accuracy = how often the model's top guess matched the player's actual move, counted only in rounds where the model was used (not the random opening).",
    rounds_per_match: rounds,
    seeds,
    players: {
      cycle: "rock, paper, scissors repeating (fully predictable)",
      biased: "60% rock, 20% paper, 20% scissors, random order (best possible accuracy 60%)",
      switching: "rock-paper-scissors cycle for the first half, reverse cycle for the second half",
      random: "uniform random control (best possible accuracy 33%)",
    },
    results,
  }, null, 2));
  console.log(`\nSaved ${outFile}`);
}

main();
