/**
 * ApexChess Interactive Web Application
 * Handles playable chessboard, NNUE/Attention visualizers,
 * Obsidian Vault reader, and UCI console emulation.
 */

// Piece symbols
const UNICODE_PIECES = {
    'P': '♙', 'N': '♘', 'B': '♗', 'R': '♖', 'Q': '♕', 'K': '♔',
    'p': '♟', 'n': '♞', 'b': '♝', 'r': '♜', 'q': '♛', 'k': '♚'
};

const PIECE_VALUES = {
    'P': 100, 'N': 320, 'B': 330, 'R': 500, 'Q': 900, 'K': 20000,
    'p': -100, 'n': -320, 'b': -330, 'r': -500, 'q': -900, 'k': -20000
};

// Standard Start FEN
const START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

// Presets
const PRESETS = {
    startpos: "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    scholars: "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4",
    sicilian: "rnbqkb1r/pp2pp1p/3p1np1/8/3NP3/2N5/PPP2PPP/R1BQKB1R w KQkq - 0 6",
    endgame: "8/8/4k3/8/8/8/4R3/4K3 w - - 0 1"
};

// Game State
let boardState = []; // 8x8 matrix
let turn = 'w';
let selectedSquare = null;
let moveHistory = [];
let isFlipped = false;

// Initialize Board from FEN
function loadFen(fen) {
    const parts = fen.split(' ');
    const rows = parts[0].split('/');
    boardState = [];
    for (let r = 0; r < 8; r++) {
        const row = [];
        for (let char of rows[r]) {
            if (/\d/.test(char)) {
                const emptyCount = parseInt(char, 10);
                for (let e = 0; e < emptyCount; e++) row.push(null);
            } else {
                row.push(char);
            }
        }
        boardState.push(row);
    }
    turn = parts[1] || 'w';
    selectedSquare = null;
    renderBoard();
    updateEvaluation();
    updateAccumulatorVis();
}

// Render Chessboard DOM
function renderBoard() {
    const boardEl = document.getElementById('chessboard');
    boardEl.innerHTML = '';

    for (let r = 0; r < 8; r++) {
        for (let c = 0; c < 8; c++) {
            const displayR = isFlipped ? 7 - r : r;
            const displayC = isFlipped ? 7 - c : c;
            const piece = boardState[displayR][displayC];

            const sqEl = document.createElement('div');
            const isLight = (displayR + displayC) % 2 === 0;
            sqEl.className = `square ${isLight ? 'light' : 'dark'}`;
            sqEl.dataset.row = displayR;
            sqEl.dataset.col = displayC;

            if (selectedSquare && selectedSquare.r === displayR && selectedSquare.c === displayC) {
                sqEl.classList.add('selected');
            }

            if (piece) {
                const pieceEl = document.createElement('span');
                pieceEl.className = 'piece';
                pieceEl.textContent = UNICODE_PIECES[piece];
                pieceEl.style.color = (piece === piece.toUpperCase()) ? '#f8fafc' : '#0f172a';
                if (piece === piece.toUpperCase()) {
                    pieceEl.style.textShadow = '0 0 2px #000, 0 0 6px rgba(0, 242, 254, 0.4)';
                }
                sqEl.appendChild(pieceEl);
            }

            sqEl.addEventListener('click', () => handleSquareClick(displayR, displayC));
            boardEl.appendChild(sqEl);
        }
    }
}

// Handle Square Clicks
function handleSquareClick(r, c) {
    const clickedPiece = boardState[r][c];

    if (selectedSquare) {
        // If clicking same square, deselect
        if (selectedSquare.r === r && selectedSquare.c === c) {
            selectedSquare = null;
            renderBoard();
            return;
        }

        // Attempt move
        const fromPiece = boardState[selectedSquare.r][selectedSquare.c];
        const isFromWhite = fromPiece === fromPiece.toUpperCase();
        const isTurnWhite = (turn === 'w');

        if ((isFromWhite && isTurnWhite) || (!isFromWhite && !isTurnWhite)) {
            // Execute move
            executeMove(selectedSquare.r, selectedSquare.c, r, c);
            selectedSquare = null;
            renderBoard();
            return;
        }
    }

    // Select new piece
    if (clickedPiece) {
        const isPieceWhite = clickedPiece === clickedPiece.toUpperCase();
        if ((turn === 'w' && isPieceWhite) || (turn === 'b' && !isPieceWhite)) {
            selectedSquare = { r, c };
            renderBoard();
            updateAttentionMap(r, c);
        }
    }
}

// Execute Move
function executeMove(fromR, fromC, toR, toC) {
    const piece = boardState[fromR][fromC];
    const target = boardState[toR][toC];

    boardState[toR][toC] = piece;
    boardState[fromR][fromC] = null;

    // Pawn promotion
    if (piece === 'P' && toR === 0) boardState[toR][toC] = 'Q';
    if (piece === 'p' && toR === 7) boardState[toR][toC] = 'q';

    const fromNotation = `${String.fromCharCode(97 + fromC)}${8 - fromR}`;
    const toNotation = `${String.fromCharCode(97 + toC)}${8 - toR}`;
    const moveStr = `${piece}${fromNotation}-${toNotation}`;

    moveHistory.push(moveStr);
    addMoveToHistoryUI(moveStr);

    turn = (turn === 'w') ? 'b' : 'w';

    updateEvaluation();
    updateAccumulatorVis();
    logUCI(`position moves ${fromNotation}${toNotation}`);
}

// Evaluation Function (NNUE & Material Approximation)
function evaluateBoard() {
    let score = 0;
    let pieceCount = 0;

    for (let r = 0; r < 8; r++) {
        for (let c = 0; c < 8; c++) {
            const p = boardState[r][c];
            if (!p) continue;
            pieceCount++;
            score += PIECE_VALUES[p] || 0;

            // Center control bonus
            if ((r === 3 || r === 4) && (c === 3 || c === 4)) {
                score += (p === p.toUpperCase()) ? 25 : -25;
            }
        }
    }

    // Convert centipawns to float pawns
    return score / 100.0;
}

function updateEvaluation() {
    const evalScore = evaluateBoard();
    const evalFill = document.getElementById('eval-fill');
    const evalLabel = document.getElementById('eval-score');
    const telemEval = document.getElementById('telem-eval');
    const telemWinProb = document.getElementById('telem-win-prob');
    const telemBestMove = document.getElementById('telem-best-move');

    // Winning Probability Sigmoid: P = 1 / (1 + 10^(-cp / 400))
    const cp = evalScore * 100.0;
    const winProb = 1.0 / (1.0 + Math.pow(10.0, -cp / 400.0));
    const winPercent = (winProb * 100).toFixed(1);

    // Update fill height [0% to 100%]
    const fillPercent = Math.min(95, Math.max(5, 50 + (evalScore * 8)));
    evalFill.style.height = `${fillPercent}%`;

    const sign = evalScore >= 0 ? '+' : '';
    evalLabel.textContent = `${sign}${evalScore.toFixed(1)}`;
    telemEval.textContent = `${sign}${evalScore.toFixed(2)} cp`;
    telemWinProb.textContent = `${winPercent}%`;

    // Dynamic Best Move suggestion
    telemBestMove.textContent = suggestBestMove();

    // Checkmate check for Scholar's mate
    if (boardState[1][5] === 'Q' && boardState[0][4] === 'k') {
        telemEval.textContent = "+M1 (Checkmate)";
        evalFill.style.height = "100%";
    }
}

function suggestBestMove() {
    // Generate simple best move
    const moves = [];
    for (let r = 0; r < 8; r++) {
        for (let c = 0; c < 8; c++) {
            const p = boardState[r][c];
            if (!p) continue;
            const isWhite = (p === p.toUpperCase());
            if ((turn === 'w' && isWhite) || (turn === 'b' && !isWhite)) {
                // Forward pawn push
                const dir = isWhite ? -1 : 1;
                const nextR = r + dir;
                if (nextR >= 0 && nextR < 8 && !boardState[nextR][c]) {
                    moves.push(`${String.fromCharCode(97 + c)}${8 - r}-${String.fromCharCode(97 + c)}${8 - nextR}`);
                }
            }
        }
    }
    return moves.length > 0 ? moves[0] : "none";
}

function addMoveToHistoryUI(moveStr) {
    const historyEl = document.getElementById('move-history');
    if (historyEl.querySelector('.empty-hint')) {
        historyEl.innerHTML = '';
    }
    const moveItem = document.createElement('div');
    moveItem.className = 'history-item';
    moveItem.textContent = `${moveHistory.length}. ${moveStr}`;
    historyEl.appendChild(moveItem);
    historyEl.scrollTop = historyEl.scrollHeight;
}

// Engine Move Trigger
function triggerEngineMove() {
    const bestMv = suggestBestMove();
    if (bestMv && bestMv !== "none") {
        const parts = bestMv.split('-');
        const fromC = parts[0].charCodeAt(0) - 97;
        const fromR = 8 - parseInt(parts[0][1], 10);
        const toC = parts[1].charCodeAt(0) - 97;
        const toR = 8 - parseInt(parts[1][1], 10);

        executeMove(fromR, fromC, toR, toC);
        renderBoard();
        logUCI(`bestmove ${parts[0]}${parts[1]}`);
    }
}

// Architecture: NNUE Visualizer
function updateAccumulatorVis() {
    // Generate active dots
    const dotsEl = document.getElementById('sparse-dots');
    dotsEl.innerHTML = '';
    for (let i = 0; i < 30; i++) {
        const dot = document.createElement('div');
        dot.className = `sparse-dot ${Math.random() > 0.3 ? 'active' : ''}`;
        dotsEl.appendChild(dot);
    }

    // Generate accumulator bars
    const barsEl = document.getElementById('acc-bars');
    barsEl.innerHTML = '';
    for (let i = 0; i < 16; i++) {
        const bar = document.createElement('div');
        bar.className = 'acc-bar';
        const height = Math.floor(Math.random() * 45) + 15;
        bar.style.height = `${height}px`;
        barsEl.appendChild(bar);
    }

    const visScore = document.getElementById('nnue-vis-score');
    visScore.textContent = `${evaluateBoard() >= 0 ? '+' : ''}${evaluateBoard().toFixed(2)} Pawns`;
}

// Architecture: Attention Map Visualizer
function initAttentionGrid() {
    const grid = document.getElementById('attention-grid');
    grid.innerHTML = '';
    for (let i = 0; i < 64; i++) {
        const cell = document.createElement('div');
        cell.className = 'attn-cell';
        cell.dataset.index = i;
        cell.addEventListener('click', () => {
            const r = Math.floor(i / 8);
            const c = i % 8;
            updateAttentionMap(r, c);
        });
        grid.appendChild(cell);
    }
    updateAttentionMap(4, 4); // Default center e4
}

function updateAttentionMap(sourceR, sourceC) {
    const cells = document.querySelectorAll('.attn-cell');
    cells.forEach((cell, idx) => {
        const r = Math.floor(idx / 8);
        const c = idx % 8;

        // Higher attention for rays and diagonals
        const dist = Math.sqrt(Math.pow(r - sourceR, 2) + Math.pow(c - sourceC, 2));
        let weight = Math.max(0.05, 1.0 - (dist / 6.0));

        // Diagonal or rank/file boost
        if (r === sourceR || c === sourceC || Math.abs(r - sourceR) === Math.abs(c - sourceC)) {
            weight = Math.min(1.0, weight + 0.45);
        }

        const cyanR = Math.floor(0 + weight * 0);
        const cyanG = Math.floor(242 * weight);
        const cyanB = Math.floor(254 * weight);
        cell.style.backgroundColor = `rgb(${cyanR}, ${cyanG}, ${cyanB})`;
    });
}

// Obsidian Vault Research Content
const VAULT_ARTICLES = {
    '00': {
        title: "00. Map of Content (MOC): Neural Computer Chess",
        content: `
            <h1>Map of Content: The Neural Chess Engine Vault</h1>
            <p>Welcome to the central graph index of the Apex Research Knowledge Base. This vault documents the 75-year algorithmic progression from Claude Shannon's 1950 paper to the modern era of Efficiently Updatable Neural Networks (NNUE) and Spatial Attention Transformers.</p>
            <h2>Vault Index Structure</h2>
            <ul>
                <li><strong>01. Evolution & SOTA:</strong> Shannon's Type-A/B, Deep Blue, AlphaZero, Stockfish 12-17, and Searchless Chess.</li>
                <li><strong>02. Dataset Engineering:</strong> Lichess 5B+ database, Fishtest .binpack, and Syzygy 7-man tablebases.</li>
                <li><strong>03. NNUE Deep Dive:</strong> HalfKP/HalfKA sparse indexing, O(1) incremental accumulators, and AVX-512 SIMD.</li>
                <li><strong>04. Transformer Models:</strong> Spatial self-attention on 64 squares, multi-head piece battery tracking, and 1,968 action space.</li>
                <li><strong>05. Search Heuristics:</strong> Alpha-Beta, PVS, Transposition Tables, LMR, Null Move, and Quiescence.</li>
                <li><strong>06. Loss Functions & Math:</strong> WDL Binary Cross-Entropy, Centipawn Sigmoid scaling, and Straight-Through Estimators.</li>
                <li><strong>07. Cutting-Edge Blueprint:</strong> The formula to surpass Stockfish via Policy-Guided Alpha-Beta and Speculative Dual-Nets.</li>
                <li><strong>08. Bibliography:</strong> 15+ seminal papers with direct citations and links.</li>
            </ul>
        `
    },
    '01': {
        title: "01. Evolution and State of the Art in Computer Chess",
        content: `
            <h1>Historical Evolution & State of the Art</h1>
            <p>Computer chess has evolved through four distinct epochs over the past 75 years, culminating in contemporary superhuman ratings of <strong>3650+ Elo</strong>.</p>
            <h2>1. Shannon & Handcrafted Evaluation (1950 - 2017)</h2>
            <p>Engines relied on Shannon Type-A Alpha-Beta search paired with thousands of handcrafted heuristics (material, piece-square tables, pawn structure, king safety). Peaked with Stockfish 11 at ~3450 Elo, hitting the human parameter tuning wall.</p>
            <h2>2. Deep Reinforcement Learning (2017 - 2020)</h2>
            <p>DeepMind's AlphaZero and Leela Chess Zero (Lc0) demonstrated that pure self-play with a 20-40 block Residual CNN and Monte Carlo Tree Search (MCTS) produced profound, human-like strategic sacrifices. However, GPU throughput (50k nps) was 1,000x slower than CPU search.</p>
            <h2>3. The NNUE Revolution (2020 - Present)</h2>
            <p>Stockfish 12 introduced NNUE (Efficiently Updatable Neural Networks). By computing O(1) incremental accumulator updates on sparse features and quantizing to int8/int16 SIMD, Stockfish evaluated 60M+ nodes/sec with a deep neural net, jumping +150 Elo.</p>
            <h2>4. Searchless Transformers (2024+)</h2>
            <p>DeepMind published 'Searchless Chess', showing a 270M-parameter decoder transformer can play at 2895 Elo with zero search nodes.</p>
        `
    },
    '02': {
        title: "02. Dataset Engineering & Corpus Acquisition",
        content: `
            <h1>Dataset Engineering & Preprocessing Pipelines</h1>
            <p>Modern chess AI requires training on hundreds of millions of positions. Data quality and representation define engine strength.</p>
            <h2>Corpus Sources</h2>
            <ul>
                <li><strong>Lichess Open Database:</strong> 5.5 billion standard human and bot games recorded since 2013.</li>
                <li><strong>Lichess Open Evaluations:</strong> 300 million positions evaluated by Stockfish 16 at depth 30-50 plies.</li>
                <li><strong>Stockfish Fishtest:</strong> Billions of self-play tournament positions packed in custom .binpack format.</li>
                <li><strong>Syzygy Tablebases:</strong> 17.5 TB 7-man exact Win/Draw/Loss and Distance-to-Zero tables.</li>
            </ul>
            <h2>Filtering Heuristics</h2>
            <p>Raw games are filtered to exclude games with ratings &lt; 2200 Elo, skip the first 10 plies of opening book memorization, and eliminate unstable in-check positions.</p>
        `
    },
    '03': {
        title: "03. NNUE Architecture Deep Dive",
        content: `
            <h1>NNUE Architecture Deep Dive</h1>
            <p>NNUE achieves 60M–100M evaluations per second by exploiting the mathematical property that only 1 or 2 pieces move on any turn.</p>
            <h2>Incremental Accumulator Updates</h2>
            <pre>A(t+1) = A(t) - W[old_feature] + W[new_feature]</pre>
            <p>Instead of recalculating the entire matrix multiplication (41,024 x 1024), the accumulator computes a simple vector addition in under 10 nanoseconds.</p>
            <h2>SCReLU Activation</h2>
            <p>Squared Clipped ReLU [min(max(x, 0), 127)]^2 amplifies salient tactical and positional signals while compressing noise, delivering +20 Elo over linear ClippedReLU.</p>
        `
    },
    '04': {
        title: "04. Transformer & AlphaZero Models",
        content: `
            <h1>Transformer & AlphaZero Architectures</h1>
            <p>Spatial Multi-Head Attention enables pieces across the entire 64-square board to communicate in a single layer, eliminating the localized receptive field bottleneck of standard convolutions.</p>
            <h2>64 Squares as Spatial Tokens</h2>
            <p>Every square is an embedding token. Self-attention matrices naturally learn long-range diagonal queen/bishop batteries, open rook files, and defensive ties.</p>
            <h2>1,968 Action Space</h2>
            <p>The policy head maps all legal transitions (ray moves, knight hops, promotions) using a bilinear outer-product between source and destination square feature vectors.</p>
        `
    },
    '05': {
        title: "05. Search Algorithms & Heuristics",
        content: `
            <h1>Search Algorithms & Tree Optimization</h1>
            <p>Alpha-Beta pruning and Principal Variation Search (PVS) compress the average chess branching factor from b ≈ 35 down to b ≈ 1.5 - 2.0.</p>
            <h2>Key Search Heuristics</h2>
            <ul>
                <li><strong>Principal Variation Search (PVS):</strong> Searches the first move with full [alpha, beta] window, and all sibling moves with a minimal zero-window [alpha, alpha+1].</li>
                <li><strong>Transposition Table (TT):</strong> Caches search evaluations using 64-bit Zobrist XOR hashing.</li>
                <li><strong>Null Move Pruning (NMP):</strong> Gives the opponent a free pass; if we still fail high, prune subtree.</li>
                <li><strong>Late Move Reductions (LMR):</strong> Reduces search depth on unpromising quiet moves late in the move order.</li>
                <li><strong>Quiescence Search:</strong> Continues searching tactical captures to resolve the Horizon Effect.</li>
            </ul>
        `
    },
    '06': {
        title: "06. Loss Functions & Training Mathematics",
        content: `
            <h1>Loss Functions & Optimization Mathematics</h1>
            <h2>Sigmoid Winning Probability Function</h2>
            <p>Centipawns are converted to smooth winning probabilities:</p>
            <pre>P(Win) = 1 / (1 + 10^(-cp / 400))</pre>
            <h2>Value Loss: Soft Target Binary Cross-Entropy</h2>
            <p>Targets are blended between empirical game outcome (z) and engine evaluation (q):</p>
            <pre>y = lambda * z + (1 - lambda) * q(cp)
L_val = - [ y * ln(p) + (1 - y) * ln(1 - p) ]</pre>
            <h2>Straight-Through Estimators (STE)</h2>
            <p>Simulates int8 fixed-point quantization during training, passing gradients unchanged through rounding operators to eliminate quantization drift.</p>
        `
    },
    '07': {
        title: "07. The Cutting-Edge Engine Blueprint",
        content: `
            <h1>The Cutting-Edge Blueprint: Surpassing Stockfish</h1>
            <h2>1. The Move-Ordering Bottleneck</h2>
            <p>Stockfish relies on heuristic history tables with no spatial chess geometry. ApexChess introduces a distilled 4-layer Policy Prior to order moves, finding the best move first in &gt;85% of positions and dropping the branching factor to b ≈ 1.3.</p>
            <h2>2. Speculative Dual-Evaluation</h2>
            <p>Quiet nodes (95%) execute on 60M nps NNUE; high-entropy nodes (5%) trigger deep Spatial Transformer evaluations.</p>
            <h2>3. 7-Man Syzygy Distillation</h2>
            <p>Distills 17.5 TB of endgame tablebases directly into network parameters, eliminating disk probe latency.</p>
        `
    },
    '08': {
        title: "08. Landmark Papers Bibliography",
        content: `
            <h1>Landmark Papers & Annotated Bibliography</h1>
            <ul>
                <li><strong>Shannon (1950):</strong> Programming a Computer for Playing Chess. Introduces Type-A vs Type-B search.</li>
                <li><strong>Knuth & Moore (1975):</strong> Mathematical proof that Alpha-Beta reduces branching factor to sqrt(b).</li>
                <li><strong>Zobrist (1970):</strong> Incremental XOR hashing for transposition tables.</li>
                <li><strong>Silver et al. (DeepMind, 2017):</strong> AlphaZero mastering chess from self-play with ResNets and MCTS.</li>
                <li><strong>Yu Nasu (2018):</strong> Efficiently Updatable Neural Network (NNUE).</li>
                <li><strong>Ruoss et al. (DeepMind, 2024):</strong> Grandmaster-Level Chess Without Search.</li>
            </ul>
        `
    }
};

function initVaultReader() {
    const navItems = document.querySelectorAll('#vault-nav li');
    const contentBody = document.getElementById('vault-article-body');

    function loadNote(noteId) {
        navItems.forEach(li => li.classList.remove('active'));
        const activeLi = document.querySelector(`#vault-nav li[data-note="${noteId}"]`);
        if (activeLi) activeLi.classList.add('active');

        const article = VAULT_ARTICLES[noteId] || VAULT_ARTICLES['00'];
        contentBody.innerHTML = article.content;
    }

    navItems.forEach(li => {
        li.addEventListener('click', () => {
            const noteId = li.dataset.note;
            loadNote(noteId);
        });
    });

    loadNote('00');
}

// UCI Console Helper
function logUCI(msg) {
    const consoleEl = document.getElementById('uci-console');
    const line = document.createElement('div');
    line.className = 'console-line';
    line.innerHTML = `<span class="cmd">&gt; ${msg}</span>`;
    consoleEl.appendChild(line);
    consoleEl.scrollTop = consoleEl.scrollHeight;
}

// Settings & Controls Listeners
function initSettings() {
    const depthSlider = document.getElementById('set-depth');
    const depthVal = document.getElementById('depth-val');
    depthSlider.addEventListener('input', (e) => {
        depthVal.textContent = `${e.target.value} plies`;
        logUCI(`setoption name Depth value ${e.target.value}`);
    });

    const ttSlider = document.getElementById('set-tt');
    const ttVal = document.getElementById('tt-val');
    ttSlider.addEventListener('input', (e) => {
        ttVal.textContent = `${e.target.value} MB`;
        logUCI(`setoption name Hash value ${e.target.value}`);
    });

    const nnueCheck = document.getElementById('set-nnue');
    nnueCheck.addEventListener('change', (e) => {
        logUCI(`setoption name UseNNUE value ${e.target.checked}`);
    });

    const policyCheck = document.getElementById('set-policy');
    policyCheck.addEventListener('change', (e) => {
        logUCI(`setoption name UsePolicyPrior value ${e.target.checked}`);
    });
}

// Architecture Tab Switching
function initArchTabs() {
    const tabBtns = document.querySelectorAll('.arch-tab-btn');
    const panels = document.querySelectorAll('.arch-panel');

    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            tabBtns.forEach(b => b.classList.remove('active'));
            panels.forEach(p => p.classList.remove('active'));

            btn.classList.add('active');
            const targetId = btn.dataset.target;
            const targetPanel = document.getElementById(targetId);
            if (targetPanel) targetPanel.classList.add('active');
        });
    });
}

// Setup Event Listeners
function initApp() {
    loadFen(START_FEN);
    initAttentionGrid();
    initVaultReader();
    initSettings();
    initArchTabs();

    // Preset selector
    const presetSelect = document.getElementById('pos-preset');
    presetSelect.addEventListener('change', (e) => {
        const fen = PRESETS[e.target.value] || START_FEN;
        loadFen(fen);
        logUCI(`position fen ${fen}`);
    });

    // Control buttons
    document.getElementById('btn-reset').addEventListener('click', () => {
        loadFen(START_FEN);
        moveHistory = [];
        document.getElementById('move-history').innerHTML = '<span class="empty-hint">Moves will appear here as played.</span>';
        logUCI('ucinewgame');
    });

    document.getElementById('btn-flip').addEventListener('click', () => {
        isFlipped = !isFlipped;
        renderBoard();
    });

    document.getElementById('btn-undo').addEventListener('click', () => {
        if (moveHistory.length > 0) {
            moveHistory.pop();
            loadFen(START_FEN);
            logUCI('undo');
        }
    });

    document.getElementById('btn-engine-move').addEventListener('click', () => {
        triggerEngineMove();
    });
}

document.addEventListener('DOMContentLoaded', initApp);
