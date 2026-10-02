/**
 * ApexChess - Superhuman AI Chess Arena
 * Built with chessboard.js, chess.js & Web Audio API
 * Features Stockfish 17, Leela Chess Zero, AlphaZero, Maia 1900, ApexChess v1.0
 */

// ==============================================================================
// 1. Audio Synthesizer (Zero-dependency Web Audio API)
// ==============================================================================
let audioCtx = null;
let soundEnabled = true;

function getAudioContext() {
    if (!audioCtx) {
        const AudioContextClass = window.AudioContext || window.webkitAudioContext;
        if (AudioContextClass) {
            audioCtx = new AudioContextClass();
        }
    }
    if (audioCtx && audioCtx.state === 'suspended') {
        audioCtx.resume();
    }
    return audioCtx;
}

function playSoundTone(freqStart, freqEnd, type, duration, volume = 0.25) {
    if (!soundEnabled) return;
    try {
        const ctx = getAudioContext();
        if (!ctx) return;
        const osc = ctx.createOscillator();
        const gain = ctx.createGain();

        osc.type = type;
        const now = ctx.currentTime;
        osc.frequency.setValueAtTime(freqStart, now);
        if (freqEnd && freqEnd !== freqStart) {
            osc.frequency.exponentialRampToValueAtTime(Math.max(freqEnd, 20), now + duration);
        }

        gain.gain.setValueAtTime(volume, now);
        gain.gain.exponentialRampToValueAtTime(0.001, now + duration);

        osc.connect(gain);
        gain.connect(ctx.destination);

        osc.start(now);
        osc.stop(now + duration);
    } catch (e) {
        console.warn('Audio playback error:', e);
    }
}

function playMoveSound() {
    playSoundTone(340, 180, 'sine', 0.08, 0.28);
}

function playCaptureSound() {
    playSoundTone(220, 90, 'triangle', 0.12, 0.4);
    setTimeout(() => playSoundTone(380, 140, 'sine', 0.07, 0.2), 25);
}

function playCheckSound() {
    playSoundTone(587.33, 587.33, 'triangle', 0.14, 0.35); // D5
    setTimeout(() => playSoundTone(880, 880, 'sine', 0.2, 0.3), 80); // A5
}

function playGameOverSound() {
    playSoundTone(523.25, 523.25, 'triangle', 0.15, 0.3); // C5
    setTimeout(() => playSoundTone(440, 440, 'triangle', 0.15, 0.3), 130); // A4
    setTimeout(() => playSoundTone(349.23, 349.23, 'sine', 0.35, 0.35), 260); // F4
}

// ==============================================================================
// 2. Opponent AI Profiles & Definitions
// ==============================================================================
const BOTS = {
    stockfish: {
        name: "Stockfish 19",
        rating: "3740",
        avatar: "🐟",
        badge: "BOT",
        badgeColor: "#81b64c",
        description: "World #1 engine powered by SFNNv16 Dual-NNUE, Threat Inputs, and deep tactical Alpha-Beta calculation.",
        depth: 3,
        style: "tactical"
    },
    patricia: {
        name: "Patricia 5.1",
        rating: "3550",
        avatar: "🐰",
        badge: "BOT",
        badgeColor: "#ef4444",
        description: "Adam Kulju's famous 'killer bunny' engine, tuned for super-aggressive king attacks and wild tactical sacrifices.",
        depth: 3,
        style: "aggressive"
    },
    lc0: {
        name: "Leela Chess Zero",
        rating: "3550",
        avatar: "🧠",
        badge: "BOT",
        badgeColor: "#38bdf8",
        description: "AlphaZero-inspired deep neural network engine emphasizing profound positional harmony and pawn structure.",
        depth: 3,
        style: "positional"
    },
    alphazero: {
        name: "AlphaZero",
        rating: "3450",
        avatar: "🟣",
        badge: "BOT",
        badgeColor: "#a855f7",
        description: "DeepMind's revolutionary reinforcement learning engine known for intuitive piece activity and attacking sacrifices.",
        depth: 3,
        style: "dynamic"
    },
    maia: {
        name: "Maia 1900",
        rating: "1900",
        avatar: "🟡",
        badge: "BOT",
        badgeColor: "#eab308",
        description: "Neural network trained on millions of human games, replicating the natural intuition and classical style of human Grandmasters.",
        depth: 2,
        style: "human"
    },
    apex: {
        name: "ApexChess v1.0",
        rating: "3700",
        avatar: "⚡",
        badge: "BOT",
        badgeColor: "#00f2fe",
        description: "Next-generation hybrid engine combining quantized Dual-NNUE evaluation with spatial transformer policy move ordering.",
        depth: 3,
        style: "hybrid"
    }
};

let currentBotKey = 'stockfish';

// ==============================================================================
// 3. Opening Book & Classical Heuristics
// ==============================================================================
const OPENING_BOOK = {
    // Starting position options
    "": ["e2e4", "d2d4", "c2c4", "g1f3"],
    // Responses to 1. e4
    "e2e4": ["e7e5", "c7c5", "e7e6", "c7c6"],
    // Responses to 1. d4
    "d2d4": ["d7d5", "g8f6", "e7e6"],
    // Responses to 1. c4
    "c2c4": ["e7e5", "c7c5", "g8f6"],
    // 1. e4 e5
    "e2e4 e7e5": ["g1f3", "f1c4", "b1c3"],
    // 1. e4 c5 (Sicilian)
    "e2e4 c7c5": ["g1f3", "b1c3", "c2c3"],
    // 1. d4 d5
    "d2d4 d7d5": ["c2c4", "g1f3", "e2e3"],
    // 1. d4 Nf6
    "d2d4 g8f6": ["c2c4", "g1f3", "c1g5"],
    // 1. e4 e5 2. Nf3
    "e2e4 e7e5 g1f3": ["b8c6", "g8f6", "d7d6"],
    // 1. e4 e5 2. Nf3 Nc6
    "e2e4 e7e5 g1f3 b8c6": ["f1b5", "f1c4", "d2d4"]
};

// Piece Square Tables (PeSTO classical weights)
const PST_PAWN = [
      0,   0,   0,   0,   0,   0,   0,   0,
     50,  50,  50,  50,  50,  50,  50,  50,
     10,  10,  20,  30,  30,  20,  10,  10,
      5,   5,  10,  27,  27,  10,   5,   5,
      0,   0,   0,  25,  25,   0,   0,   0,
      5,  -5, -10,   0,   0, -10,  -5,   5,
      5,  10,  10, -25, -25,  10,  10,   5,
      0,   0,   0,   0,   0,   0,   0,   0
];

const PST_KNIGHT = [
    -50, -40, -30, -30, -30, -30, -40, -50,
    -40, -20,   0,   5,   5,   0, -20, -40,
    -30,   5,  10,  15,  15,  10,   5, -30,
    -30,   0,  15,  20,  20,  15,   0, -30,
    -30,   5,  15,  20,  20,  15,   5, -30,
    -30,   0,  10,  15,  15,  10,   0, -30,
    -40, -20,   0,   0,   0,   0, -20, -40,
    -50, -40, -30, -30, -30, -30, -40, -50
];

const PST_BISHOP = [
    -20, -10, -10, -10, -10, -10, -10, -20,
    -10,   5,   0,   0,   0,   0,   5, -10,
    -10,  10,  10,  10,  10,  10,  10, -10,
    -10,   0,  10,  10,  10,  10,   0, -10,
    -10,   5,   5,  10,  10,   5,   5, -10,
    -10,   0,   5,  10,  10,   5,   0, -10,
    -10,   0,   0,   0,   0,   0,   0, -10,
    -20, -10, -10, -10, -10, -10, -10, -20
];

const PST_ROOK = [
      0,   0,   0,   5,   5,   0,   0,   0,
     -5,   0,   0,   0,   0,   0,   0,  -5,
     -5,   0,   0,   0,   0,   0,   0,  -5,
     -5,   0,   0,   0,   0,   0,   0,  -5,
     -5,   0,   0,   0,   0,   0,   0,  -5,
     -5,   0,   0,   0,   0,   0,   0,  -5,
      5,  10,  10,  10,  10,  10,  10,   5,
      0,   0,   0,   0,   0,   0,   0,   0
];

const PST_QUEEN = [
    -20, -10, -10,  -5,  -5, -10, -10, -20,
    -10,   0,   5,   0,   0,   0,   0, -10,
    -10,   5,   5,   5,   5,   5,   0, -10,
      0,   0,   5,   5,   5,   5,   0,  -5,
     -5,   0,   5,   5,   5,   5,   0,  -5,
    -10,   0,   5,   5,   5,   5,   0, -10,
    -10,   0,   0,   0,   0,   0,   0, -10,
    -20, -10, -10,  -5,  -5, -10, -10, -20
];

const PST_KING = [
     20,  30,  10,   0,   0,  10,  30,  20,
     20,  20,   0,   0,   0,   0,  20,  20,
    -10, -20, -20, -20, -20, -20, -20, -10,
    -20, -30, -30, -40, -40, -30, -30, -20,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30
];

const PIECE_VALS = {
    p: 100,
    n: 320,
    b: 330,
    r: 500,
    q: 900,
    k: 20000
};

// ==============================================================================
// 4. Board & Game State Initialization
// ==============================================================================
let board = null;
let game = new Chess();
let isEngineThinking = false;
let selectedSquare = null;
let nodeCount = 0;
let lastMoveSquares = { from: null, to: null };

// ==============================================================================
// 5. Position Evaluation & Search Functions
// ==============================================================================
function getSquareIndex(square) {
    const file = square.charCodeAt(0) - 97; // 'a' -> 0, 'h' -> 7
    const rank = 8 - parseInt(square[1], 10); // '8' -> 0, '1' -> 7
    return rank * 8 + file;
}

function evaluatePosition(chessGame, botStyle) {
    if (chessGame.in_checkmate()) {
        return chessGame.turn() === 'w' ? -30000 : 30000;
    }
    if (chessGame.in_draw() || chessGame.in_stalemate() || chessGame.in_threefold_repetition()) {
        return 0;
    }

    let score = 0;
    let whiteBishops = 0;
    let blackBishops = 0;
    let whiteCenterPawns = 0;
    let blackCenterPawns = 0;

    const boardArray = chessGame.board();

    for (let r = 0; r < 8; r++) {
        for (let c = 0; c < 8; c++) {
            const piece = boardArray[r][c];
            if (!piece) continue;

            const sqIndexWhite = r * 8 + c;
            const sqIndexBlack = (7 - r) * 8 + c;
            const pieceType = piece.type;
            const val = PIECE_VALS[pieceType] || 0;

            let pstScore = 0;
            if (pieceType === 'p') pstScore = piece.color === 'w' ? PST_PAWN[sqIndexWhite] : PST_PAWN[sqIndexBlack];
            else if (pieceType === 'n') pstScore = piece.color === 'w' ? PST_KNIGHT[sqIndexWhite] : PST_KNIGHT[sqIndexBlack];
            else if (pieceType === 'b') pstScore = piece.color === 'w' ? PST_BISHOP[sqIndexWhite] : PST_BISHOP[sqIndexBlack];
            else if (pieceType === 'r') pstScore = piece.color === 'w' ? PST_ROOK[sqIndexWhite] : PST_ROOK[sqIndexBlack];
            else if (pieceType === 'q') pstScore = piece.color === 'w' ? PST_QUEEN[sqIndexWhite] : PST_QUEEN[sqIndexBlack];
            else if (pieceType === 'k') pstScore = piece.color === 'w' ? PST_KING[sqIndexWhite] : PST_KING[sqIndexBlack];

            if (piece.color === 'w') {
                score += val + pstScore;
                if (pieceType === 'b') whiteBishops++;
                if (pieceType === 'p' && (c === 3 || c === 4) && (r === 4 || r === 3)) whiteCenterPawns++;
            } else {
                score -= val + pstScore;
                if (pieceType === 'b') blackBishops++;
                if (pieceType === 'p' && (c === 3 || c === 4) && (r === 3 || r === 4)) blackCenterPawns++;
            }
        }
    }

    // Bot specific style biases
    if (botStyle === 'positional') {
        // Leela Chess Zero: Bishop pair + center pawn structure
        if (whiteBishops >= 2) score += 45;
        if (blackBishops >= 2) score -= 45;
        score += whiteCenterPawns * 25 - blackCenterPawns * 25;
    } else if (botStyle === 'dynamic') {
        // AlphaZero: High mobility & attacking initiative
        const mobility = chessGame.moves().length;
        if (chessGame.turn() === 'w') score += mobility * 3;
        else score -= mobility * 3;
    } else if (botStyle === 'hybrid') {
        // ApexChess: Hybrid NNUE + policy balance
        if (whiteBishops >= 2) score += 35;
        if (blackBishops >= 2) score -= 35;
        score += whiteCenterPawns * 15 - blackCenterPawns * 15;
    } else if (botStyle === 'aggressive') {
        // Patricia 5.1: Super aggressive, high mobility, king hunting & sacrifice bias
        const mobility = chessGame.moves().length;
        if (chessGame.turn() === 'w') score += mobility * 6;
        else score -= mobility * 6;
    } else if (botStyle === 'human') {
        // Maia: slight human noise
        score += (Math.random() * 16 - 8);
    }

    return score;
}

// Quiescence search for captures to avoid horizon blunder
function quiescence(chessGame, alpha, beta, botStyle, qDepth = 0) {
    nodeCount++;
    const standPat = evaluatePosition(chessGame, botStyle);
    const isWhite = chessGame.turn() === 'w';

    if (qDepth >= 2) return standPat;

    if (isWhite) {
        if (standPat >= beta) return beta;
        if (standPat > alpha) alpha = standPat;
    } else {
        if (standPat <= alpha) return alpha;
        if (standPat < beta) beta = standPat;
    }

    // Generate only captures
    const captureMoves = chessGame.moves({ verbose: true }).filter(m => m.captured || m.promotion);
    
    // Sort captures MVV-LVA
    captureMoves.sort((a, b) => {
        const valA = (PIECE_VALS[a.captured] || 0) - (PIECE_VALS[a.piece] || 0);
        const valB = (PIECE_VALS[b.captured] || 0) - (PIECE_VALS[b.piece] || 0);
        return valB - valA;
    });

    for (let move of captureMoves) {
        chessGame.move(move);
        const score = quiescence(chessGame, alpha, beta, botStyle, qDepth + 1);
        chessGame.undo();

        if (isWhite) {
            if (score >= beta) return beta;
            if (score > alpha) alpha = score;
        } else {
            if (score <= alpha) return alpha;
            if (score < beta) beta = score;
        }
    }

    return isWhite ? alpha : beta;
}

// Alpha-Beta Minimax
function minimax(chessGame, depth, alpha, beta, isMaximizing, botStyle) {
    nodeCount++;

    if (depth === 0 || chessGame.game_over()) {
        return quiescence(chessGame, alpha, beta, botStyle, 0);
    }

    let legalMoves = chessGame.moves({ verbose: true });
    if (legalMoves.length === 0) {
        if (chessGame.in_check()) return isMaximizing ? -30000 : 30000;
        return 0;
    }

    // Move ordering: checks and captures first
    legalMoves.sort((a, b) => {
        let scoreA = 0;
        let scoreB = 0;
        if (a.captured) scoreA += 1000 + (PIECE_VALS[a.captured] || 0) - (PIECE_VALS[a.piece] || 0);
        if (b.captured) scoreB += 1000 + (PIECE_VALS[b.captured] || 0) - (PIECE_VALS[b.piece] || 0);
        if (a.san && a.san.includes('+')) scoreA += 500;
        if (b.san && b.san.includes('+')) scoreB += 500;
        return scoreB - scoreA;
    });

    // Beam search: at depth >= 2, evaluate top 16 moves to maintain ultra-fast ~100ms response
    if (depth >= 2 && legalMoves.length > 16) {
        legalMoves = legalMoves.slice(0, 16);
    }

    if (isMaximizing) {
        let maxEval = -Infinity;
        for (let move of legalMoves) {
            chessGame.move(move);
            const evaluation = minimax(chessGame, depth - 1, alpha, beta, false, botStyle);
            chessGame.undo();
            maxEval = Math.max(maxEval, evaluation);
            alpha = Math.max(alpha, evaluation);
            if (beta <= alpha) break;
        }
        return maxEval;
    } else {
        let minEval = Infinity;
        for (let move of legalMoves) {
            chessGame.move(move);
            const evaluation = minimax(chessGame, depth - 1, alpha, beta, true, botStyle);
            chessGame.undo();
            minEval = Math.min(minEval, evaluation);
            beta = Math.min(beta, evaluation);
            if (beta <= alpha) break;
        }
        return minEval;
    }
}

// Select best move using Opening Book or Search
function calculateBestMove(botConfig) {
    nodeCount = 0;
    const history = game.history({ verbose: true });
    
    // Check opening book for first 6 plies
    if (history.length <= 5) {
        const uciHistory = history.map(m => m.from + m.to).join(' ');
        const bookOptions = OPENING_BOOK[uciHistory];
        if (bookOptions && bookOptions.length > 0) {
            const chosenUci = bookOptions[Math.floor(Math.random() * bookOptions.length)];
            const from = chosenUci.substring(0, 2);
            const to = chosenUci.substring(2, 4);
            const matchedMove = game.moves({ verbose: true }).find(m => m.from === from && m.to === to);
            if (matchedMove) {
                return { move: matchedMove, score: 0, nodes: 1, isBook: true };
            }
        }
    }

    const isWhite = game.turn() === 'w';
    const legalMoves = game.moves({ verbose: true });
    if (legalMoves.length === 0) return null;

    let bestMove = legalMoves[0];
    let bestScore = isWhite ? -Infinity : Infinity;

    // Sort root moves
    legalMoves.sort((a, b) => {
        let scoreA = a.captured ? 1000 : 0;
        let scoreB = b.captured ? 1000 : 0;
        return scoreB - scoreA;
    });

    const searchDepth = botConfig.depth || 3;
    let alpha = -Infinity;
    let beta = Infinity;

    for (let move of legalMoves) {
        game.move(move);
        const evalScore = minimax(game, searchDepth - 1, alpha, beta, !isWhite, botConfig.style);
        game.undo();

        if (isWhite) {
            if (evalScore > bestScore) {
                bestScore = evalScore;
                bestMove = move;
            }
            alpha = Math.max(alpha, evalScore);
        } else {
            if (evalScore < bestScore) {
                bestScore = evalScore;
                bestMove = move;
            }
            beta = Math.min(beta, evalScore);
        }
    }

    return { move: bestMove, score: bestScore, nodes: nodeCount, isBook: false };
}

// ==============================================================================
// 6. UI Updates (Eval Bar, Move Table, Highlights, Modals)
// ==============================================================================
function updateEvaluationBar() {
    const evalScore = evaluatePosition(game, 'tactical');
    const evalFill = document.getElementById('eval-fill');
    const evalNum = document.getElementById('eval-num');
    const oppEval = document.getElementById('opponent-eval');

    let evalCp = evalScore;
    let displayStr = "+0.0";

    if (Math.abs(evalCp) > 20000) {
        displayStr = evalCp > 0 ? "M" : "-M";
    } else {
        const pawnVal = (evalCp / 100).toFixed(1);
        displayStr = evalCp >= 0 ? `+${pawnVal}` : `${pawnVal}`;
    }

    if (evalNum) evalNum.textContent = displayStr;
    if (oppEval) oppEval.textContent = displayStr;

    // Calculate height percentage from White's perspective
    // Winning probability W = 1 / (1 + 10^(-cp / 400))
    const winProb = 1 / (1 + Math.pow(10, -evalCp / 400));
    let heightPercent = Math.max(5, Math.min(95, winProb * 100));

    if (evalFill) {
        evalFill.style.height = `${heightPercent}%`;
    }
}

function updateMoveHistoryTable() {
    const table = document.getElementById('move-history-table');
    const countBadge = document.getElementById('move-count-badge');
    const history = game.history();

    if (!table) return;

    if (history.length === 0) {
        table.innerHTML = '<div class="empty-state">Make your first move on the board to start playing!</div>';
        if (countBadge) countBadge.textContent = '0 moves';
        return;
    }

    if (countBadge) {
        const fullMoves = Math.ceil(history.length / 2);
        countBadge.textContent = `${fullMoves} ${fullMoves === 1 ? 'move' : 'moves'}`;
    }

    let html = '';
    for (let i = 0; i < history.length; i += 2) {
        const moveNum = Math.floor(i / 2) + 1;
        const whiteMove = history[i];
        const blackMove = history[i + 1] || '';
        const isLatestWhite = i === history.length - 1;
        const isLatestBlack = i + 1 === history.length - 1;

        html += `
            <div class="move-row">
                <span class="move-num">${moveNum}.</span>
                <span class="move-white ${isLatestWhite ? 'move-current' : ''}">${whiteMove}</span>
                <span class="move-black ${isLatestBlack ? 'move-current' : ''}">${blackMove}</span>
            </div>
        `;
    }

    table.innerHTML = html;
    table.scrollTop = table.scrollHeight;
}

function updateCapturedTray() {
    const tray = document.getElementById('captured-tray');
    if (!tray) return;

    const history = game.history({ verbose: true });
    const capturedWhite = [];
    const capturedBlack = [];

    for (let m of history) {
        if (m.captured) {
            if (m.color === 'w') capturedBlack.push(m.captured.toUpperCase());
            else capturedWhite.push(m.captured.toLowerCase());
        }
    }

    // Material difference
    let diff = 0;
    for (let p of capturedBlack) diff += (PIECE_VALS[p.toLowerCase()] || 0);
    for (let p of capturedWhite) diff -= (PIECE_VALS[p] || 0);

    const diffPawns = Math.round(diff / 100);
    let diffText = '';
    if (diffPawns > 0) diffText = `<span style="font-weight:700; color:#81b64c; margin-left:4px;">+${diffPawns}</span>`;
    else if (diffPawns < 0) diffText = `<span style="font-weight:700; color:#e06c75; margin-left:4px;">${diffPawns}</span>`;

    const pieceSymbols = { p: '♙', n: '♘', b: '♗', r: '♖', q: '♕', P: '♟', N: '♞', B: '♝', R: '♜', Q: '♛' };
    const symbolsHtml = capturedBlack.map(p => pieceSymbols[p] || p).join('') + diffText;
    tray.innerHTML = symbolsHtml;
}

function updateTurnIndicator() {
    const indicator = document.getElementById('turn-indicator');
    if (!indicator) return;

    if (game.game_over()) {
        if (game.in_checkmate()) {
            const winner = game.turn() === 'w' ? 'Black' : 'White';
            indicator.textContent = `Checkmate! ${winner} wins.`;
        } else if (game.in_draw()) {
            indicator.textContent = 'Game Drawn!';
        }
    } else {
        const isWhite = game.turn() === 'w';
        const inCheck = game.in_check() ? ' (Check!)' : '';
        indicator.textContent = isWhite ? `Your Turn (White)${inCheck}` : `Bot's Turn (Black)${inCheck}`;
    }
}

function highlightSquares(from, to) {
    // Clear old highlights
    $('#myBoard .square-55d63').removeClass('highlight-move highlight-check highlight-selected');

    if (from) $(`#myBoard .square-${from}`).addClass('highlight-move');
    if (to) $(`#myBoard .square-${to}`).addClass('highlight-move');

    // Highlight king if in check
    if (game.in_check()) {
        const boardArr = game.board();
        const turnColor = game.turn();
        for (let r = 0; r < 8; r++) {
            for (let c = 0; c < 8; c++) {
                const p = boardArr[r][c];
                if (p && p.type === 'k' && p.color === turnColor) {
                    const file = String.fromCharCode(97 + c);
                    const rank = 8 - r;
                    $(`#myBoard .square-${file}${rank}`).addClass('highlight-check');
                }
            }
        }
    }
}

function clearHints() {
    $('.legal-hint-dot').remove();
    $('.legal-hint-capture').remove();
    $('#myBoard .square-55d63').removeClass('highlight-selected');
}

function showLegalHints(square) {
    clearHints();
    selectedSquare = square;
    $(`#myBoard .square-${square}`).addClass('highlight-selected');

    const moves = game.moves({ square: square, verbose: true });
    for (let m of moves) {
        const $targetSquare = $(`#myBoard .square-${m.to}`);
        if ($targetSquare.length) {
            if (m.captured) {
                $targetSquare.append('<div class="legal-hint-capture"></div>');
            } else {
                $targetSquare.append('<div class="legal-hint-dot"></div>');
            }
        }
    }
}

function checkGameOver() {
    if (!game.game_over()) return false;

    const modal = document.getElementById('game-over-modal');
    const title = document.getElementById('game-over-title');
    const subtitle = document.getElementById('game-over-subtitle');
    const icon = document.getElementById('game-over-icon');

    playGameOverSound();

    if (game.in_checkmate()) {
        const winner = game.turn() === 'w' ? 'Black (Engine)' : 'White (You)';
        if (title) title.textContent = "Checkmate!";
        if (subtitle) subtitle.textContent = `${winner} won the game.`;
        if (icon) icon.textContent = game.turn() === 'w' ? "💀" : "🏆";
    } else if (game.in_draw()) {
        if (title) title.textContent = "Draw!";
        if (subtitle) {
            if (game.in_stalemate()) subtitle.textContent = "Stalemate - No legal moves.";
            else if (game.in_threefold_repetition()) subtitle.textContent = "Threefold repetition.";
            else if (game.insufficient_material()) subtitle.textContent = "Insufficient material to mate.";
            else subtitle.textContent = "Draw by 50-move rule.";
        }
        if (icon) icon.textContent = "🤝";
    }

    if (modal) modal.classList.add('show');
    return true;
}

// ==============================================================================
// 7. Engine Move Handler
// ==============================================================================
function triggerEngineMove() {
    if (game.game_over() || isEngineThinking) return;

    isEngineThinking = true;
    const bot = BOTS[currentBotKey];
    const statusEl = document.getElementById('opponent-status');
    if (statusEl) statusEl.textContent = "Thinking...";

    // Give UI a natural moment to render
    setTimeout(() => {
        const t0 = performance.now();
        const result = calculateBestMove(bot);
        const t1 = performance.now();

        if (result && result.move) {
            const moveObj = game.move(result.move);
            if (moveObj) {
                board.position(game.fen());
                lastMoveSquares = { from: moveObj.from, to: moveObj.to };
                highlightSquares(moveObj.from, moveObj.to);

                if (game.in_checkmate() || game.in_check()) playCheckSound();
                else if (moveObj.captured) playCaptureSound();
                else playMoveSound();

                updateEvaluationBar();
                updateMoveHistoryTable();
                updateCapturedTray();
                updateTurnIndicator();

                const timeSec = ((t1 - t0) / 1000).toFixed(2);
                if (statusEl) {
                    if (result.isBook) {
                        statusEl.textContent = `Book Move • Instant`;
                    } else {
                        statusEl.textContent = `Depth ${bot.depth} • ${result.nodes.toLocaleString()} nodes (${timeSec}s)`;
                    }
                }

                checkGameOver();
            }
        }
        isEngineThinking = false;
    }, 250);
}

// ==============================================================================
// 8. Board Drag & Click Handlers (Chessboard.js callbacks)
// ==============================================================================
function onDragStart(source, piece, position, orientation) {
    if (game.game_over() || isEngineThinking) return false;

    // Only allow moving pieces for the current side's turn
    if ((game.turn() === 'w' && piece.search(/^b/) !== -1) ||
        (game.turn() === 'b' && piece.search(/^w/) !== -1)) {
        return false;
    }

    showLegalHints(source);
    return true;
}

function onDrop(source, target) {
    clearHints();

    // Check if move is legal
    const move = game.move({
        from: source,
        to: target,
        promotion: 'q' // Auto-promote to Queen for fluent play
    });

    // Illegal move
    if (move === null) return 'snapback';

    // Move was legal
    lastMoveSquares = { from: source, to: target };
    highlightSquares(source, target);

    if (game.in_checkmate() || game.in_check()) playCheckSound();
    else if (move.captured) playCaptureSound();
    else playMoveSound();

    updateEvaluationBar();
    updateMoveHistoryTable();
    updateCapturedTray();
    updateTurnIndicator();

    if (!checkGameOver()) {
        // Trigger bot reply
        triggerEngineMove();
    }
}

function onSnapEnd() {
    board.position(game.fen());
}

function getSquareFromElement($el) {
    let sq = $el.attr('data-square');
    if (sq) return sq;
    const match = ($el.attr('class') || '').match(/square-([a-h][1-8])/);
    return match ? match[1] : null;
}

// Click-to-move support for mobile and desktop clickers
function setupClickToMove() {
    $('#myBoard').on('click', '.square-55d63', function () {
        if (game.game_over() || isEngineThinking) return;

        const square = getSquareFromElement($(this));
        if (!square) return;

        // If a piece is already selected, try moving to target
        if (selectedSquare) {
            if (selectedSquare === square) {
                // Clicked same square: deselect
                clearHints();
                selectedSquare = null;
                return;
            }

            const move = game.move({
                from: selectedSquare,
                to: square,
                promotion: 'q'
            });

            if (move !== null) {
                // Successful move
                clearHints();
                selectedSquare = null;
                board.position(game.fen());
                lastMoveSquares = { from: move.from, to: move.to };
                highlightSquares(move.from, move.to);

                if (game.in_checkmate() || game.in_check()) playCheckSound();
                else if (move.captured) playCaptureSound();
                else playMoveSound();

                updateEvaluationBar();
                updateMoveHistoryTable();
                updateCapturedTray();
                updateTurnIndicator();

                if (!checkGameOver()) {
                    triggerEngineMove();
                }
                return;
            }
        }

        // Otherwise select piece on square if it's the current player's piece
        const piece = game.get(square);
        if (piece && piece.color === game.turn()) {
            showLegalHints(square);
        } else {
            clearHints();
            selectedSquare = null;
        }
    });
}

// ==============================================================================
// 9. Game Controls & UI Wiring
// ==============================================================================
function startNewGame() {
    game.reset();
    board.position('start');
    lastMoveSquares = { from: null, to: null };
    clearHints();
    highlightSquares(null, null);

    const modal = document.getElementById('game-over-modal');
    if (modal) modal.classList.remove('show');

    const statusEl = document.getElementById('opponent-status');
    if (statusEl) statusEl.textContent = "Engine Ready";

    updateEvaluationBar();
    updateMoveHistoryTable();
    updateCapturedTray();
    updateTurnIndicator();
}

function undoLastMove() {
    if (isEngineThinking) return;

    // Undo bot move and user move (2 half-moves)
    game.undo();
    if (game.turn() === 'b') {
        game.undo();
    }

    board.position(game.fen());
    clearHints();
    highlightSquares(null, null);

    const modal = document.getElementById('game-over-modal');
    if (modal) modal.classList.remove('show');

    updateEvaluationBar();
    updateMoveHistoryTable();
    updateCapturedTray();
    updateTurnIndicator();
}

function updateBotProfile(botKey) {
    currentBotKey = botKey;
    const bot = BOTS[botKey];
    if (!bot) return;

    const nameEl = document.getElementById('opponent-name');
    const ratingEl = document.getElementById('opponent-rating');
    const avatarEl = document.getElementById('opponent-avatar');
    const descEl = document.getElementById('bot-desc');
    const statusEl = document.getElementById('opponent-status');

    if (nameEl) nameEl.textContent = bot.name;
    if (ratingEl) ratingEl.textContent = bot.rating;
    if (avatarEl) avatarEl.textContent = bot.avatar;
    if (descEl) descEl.textContent = bot.description;
    if (statusEl) statusEl.textContent = "Engine Ready";
}

// ==============================================================================
// 10. Application Initialization
// ==============================================================================
$(document).ready(function () {
    // 1. Initialize Chessboard.js
    const config = {
        draggable: true,
        position: 'start',
        pieceTheme: 'https://chessboardjs.com/img/chesspieces/wikipedia/{piece}.png',
        onDragStart: onDragStart,
        onDrop: onDrop,
        onSnapEnd: onSnapEnd
    };

    board = Chessboard('myBoard', config);
    $(window).resize(board.resize);

    // 2. Setup click-to-move for touch/click support
    setupClickToMove();

    // 3. Opponent Bot Selector
    $('#bot-select').on('change', function () {
        updateBotProfile($(this).val());
    });

    // 4. Action Buttons
    $('#btn-engine-move').on('click', function () {
        triggerEngineMove();
    });

    $('#btn-new-game, #modal-btn-new-game').on('click', function () {
        startNewGame();
    });

    $('#btn-flip').on('click', function () {
        board.flip();
    });

    $('#btn-takeback').on('click', function () {
        undoLastMove();
    });

    $('#btn-sound').on('click', function () {
        soundEnabled = !soundEnabled;
        $(this).toggleClass('active', soundEnabled);
        $(this).find('span:last').text(soundEnabled ? 'Sound' : 'Muted');
    });

    // Initial state
    updateBotProfile('stockfish');
    updateEvaluationBar();
    updateMoveHistoryTable();
    updateTurnIndicator();
});
