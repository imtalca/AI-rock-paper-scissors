// Same match script as play.js, but the AI is the ATTENTION model from index.html
// (move + position one-hot input, Q/K/V self-attention). State is kept in
// state-attention.json so the LSTM match state is left alone.
//
//   node play_attention.js rock|paper|scissors   play a round
//   node play_attention.js status                show score and history
//   node play_attention.js reset                 start a new match (fresh, untrained model)

const fs = require("fs");
const path = require("path");
const tf = require("@tensorflow/tfjs");

const STATE_FILE = path.join(__dirname, "state-attention.json");

// ---- Copied from index.html ------------------------------------------------

const MOVES = ["rock", "paper", "scissors"];
const WINDOW = 6;
const COUNTER = { rock: "paper", paper: "scissors", scissors: "rock" };

function randomAIMove() {
  return MOVES[Math.floor(Math.random() * MOVES.length)];
}

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

function encodeWindow(moves) {
  return moves.map((m, pos) => oneHotWithPosition(m, pos));
}

function buildDataset(history) {
  if (history.length <= WINDOW) return null;
  const xArrays = [];
  const yArrays = [];
  for (let i = WINDOW; i < history.length; i++) {
    xArrays.push(encodeWindow(history.slice(i - WINDOW, i)));
    yArrays.push(oneHot(history[i]));
  }
  return { xs: tf.tensor3d(xArrays), ys: tf.tensor2d(yArrays) };
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

// Returns the AI's move plus what it expected (null probs = random fallback).
function smartAIMove(model, history) {
  if (history.length < WINDOW || model === null) {
    return { aiMove: randomAIMove(), predicted: null, probs: null };
  }
  const input = tf.tensor3d([encodeWindow(history.slice(-WINDOW))]);
  const output = model.predict(input);
  const probs = Array.from(output.dataSync());
  input.dispose();
  output.dispose();

  let best = 0;
  for (let i = 1; i < probs.length; i++) if (probs[i] > probs[best]) best = i;
  const predicted = MOVES[best];
  return { aiMove: COUNTER[predicted], predicted, probs };
}

async function trainModel(model, history) {
  const data = buildDataset(history);
  if (data === null) return model;
  if (model === null) model = createAttentionModel();
  try {
    await model.fit(data.xs, data.ys, { epochs: 20, shuffle: true, verbose: 0 });
  } finally {
    data.xs.dispose();
    data.ys.dispose();
  }
  return model;
}

// ---- Saving / loading state between runs -----------------------------------

function freshState() {
  return { history: [], rounds: [], score: { you: 0, ai: 0, ties: 0 }, model: null };
}

function loadState() {
  if (!fs.existsSync(STATE_FILE)) return freshState();
  return JSON.parse(fs.readFileSync(STATE_FILE, "utf8"));
}

async function serializeModel(model) {
  if (model === null) return null;
  const weights = model.getWeights().map(w => ({ shape: w.shape, data: Array.from(w.dataSync()) }));
  const optWeights = await model.optimizer.getWeights();
  const optimizer = optWeights.map(w => ({
    name: w.name, shape: w.tensor.shape, data: Array.from(w.tensor.dataSync()),
  }));
  return { weights, optimizer };
}

async function restoreModel(saved) {
  if (saved === null) return null;
  const model = createAttentionModel();
  model.setWeights(saved.weights.map(w => tf.tensor(w.data, w.shape)));
  await model.optimizer.setWeights(
    saved.optimizer.map(w => ({ name: w.name, tensor: tf.tensor(w.data, w.shape) }))
  );
  return model;
}

function saveState(state) {
  fs.writeFileSync(STATE_FILE, JSON.stringify(state));
}

// ---- Commands ---------------------------------------------------------------

function pct(p) {
  return `rock ${(p[0] * 100).toFixed(0)}%  paper ${(p[1] * 100).toFixed(0)}%  scissors ${(p[2] * 100).toFixed(0)}%`;
}

function printScore(state) {
  const { you, ai, ties } = state.score;
  console.log(`Round ${state.rounds.length}   You: ${you}   AI: ${ai}   Ties: ${ties}`);
}

async function playRound(yourMove) {
  const state = loadState();
  let model = await restoreModel(state.model);

  // AI commits first, exactly like the click handler in index.html.
  const { aiMove, predicted, probs } = smartAIMove(model, state.history);
  state.history.push(yourMove);
  const winner = decideWinner(yourMove, aiMove);
  if (winner === "you") state.score.you++;
  else if (winner === "ai") state.score.ai++;
  else state.score.ties++;
  state.rounds.push({ you: yourMove, ai: aiMove, winner, predicted, probs });

  model = await trainModel(model, state.history);
  state.model = await serializeModel(model);
  saveState(state);

  const resultText =
    winner === "tie" ? `Tie! Both played ${yourMove}.`
    : winner === "you" ? `You win! ${yourMove} beats ${aiMove}.`
    : `AI wins! ${aiMove} beats ${yourMove}.`;
  console.log(resultText);
  // `blind` as a second argument hides the AI's expectation and probabilities.
  if (process.argv[3] !== "blind") {
    console.log(probs ? `AI expected you to play ${predicted}  (${pct(probs)})` : "AI played randomly (no model yet)");
  }
  printScore(state);
}

function showStatus() {
  const state = loadState();
  printScore(state);
  const letter = m => m[0].toUpperCase();
  console.log("You: " + state.rounds.map(r => letter(r.you)).join(""));
  console.log("AI:  " + state.rounds.map(r => letter(r.ai)).join(""));
  console.log("Win: " + state.rounds.map(r => ({ you: "+", ai: "-", tie: "=" })[r.winner]).join(""));
}

async function main() {
  await tf.setBackend("cpu");
  const arg = (process.argv[2] || "").toLowerCase();
  if (MOVES.includes(arg)) await playRound(arg);
  else if (arg === "status") showStatus();
  else if (arg === "reset") { saveState(freshState()); console.log("New match started."); }
  else console.log("Usage: node play_attention.js rock|paper|scissors|status|reset");
}

main();
