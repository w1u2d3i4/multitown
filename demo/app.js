"use strict";

const replay = [
  {
    clock: "00:00",
    headline: "PRIORITY TASK INBOUND",
    detail: "Both organizations receive the same work order.",
    a4State: "READY",
    a8State: "READY",
    a4Event: "Awaiting work order…",
    a8Event: "Awaiting work order…",
  },
  {
    clock: "00:01",
    headline: "ORGANIZATIONS CHOOSE A ROUTE",
    detail: "A4 broadcasts; A8 reads budget and confidence state first.",
    a4State: "BROADCAST",
    a8State: "ROUTING",
    a4Event: "delegate → worker + specialist + review",
    a8Event: "route → worker (low-cost first pass)",
  },
  {
    clock: "00:02",
    headline: "PARALLEL SPEND VS TARGETED WORK",
    detail: "Fixed delegation activates every specialist; A8 starts with one worker.",
    a4State: "3 ACTIVE",
    a8State: "1 ACTIVE",
    a4Event: "three model calls consuming budget",
    a8Event: "worker draft completed within budget",
  },
  {
    clock: "00:03",
    headline: "VALIDATION CATCHES UNCERTAINTY",
    detail: "The adaptive town detects weak disagreement before delivery.",
    a4State: "QUEUEING",
    a8State: "FLAGGED",
    a4Event: "review queue growing…",
    a8Event: "validator → disagreement detected",
    a8Alert: true,
  },
  {
    clock: "00:04",
    headline: "A8 ESCALATES ONLY WHEN NEEDED",
    detail: "One expert resolves the flagged case while A4 waits on all branches.",
    a4State: "WAITING",
    a8State: "ESCALATE",
    a4Event: "waiting for mandatory specialist review",
    a8Event: "expert → targeted correction",
  },
  {
    clock: "00:05",
    headline: "FINAL SAFETY GATE",
    detail: "The corrected result passes validation and exits the adaptive town.",
    a4State: "MERGING",
    a8State: "VERIFIED",
    a4Event: "merging redundant responses",
    a8Event: "stop → validated result delivered",
  },
  {
    clock: "00:06",
    headline: "A8 WINS THE FROZEN COMPARISON",
    detail: "+11.67 pp success with 76.54% fewer tokens than A4.",
    a4State: "COMPLETE",
    a8State: "WINNER",
    a4Event: "67.22% success · token index 100%",
    a8Event: "78.89% success · token index 23.46%",
    complete: true,
  },
];

const field = (name) => document.querySelector(`[data-field="${name}"]`);
const towns = {
  a4: document.querySelector('[data-town="a4"]'),
  a8: document.querySelector('[data-town="a8"]'),
};

let step = 0;
let playing = true;
let timer = null;

function render(nextStep) {
  step = Math.max(0, Math.min(replay.length - 1, nextStep));
  const scene = replay[step];

  field("clock").textContent = scene.clock;
  field("headline").textContent = scene.headline;
  field("detail").textContent = scene.detail;
  field("a4-state").textContent = scene.a4State;
  field("a8-state").textContent = scene.a8State;
  field("a4-event").textContent = scene.a4Event;
  field("a8-event").textContent = scene.a8Event;
  field("timeline-fill").style.width = `${(step / (replay.length - 1)) * 100}%`;

  for (const [name, town] of Object.entries(towns)) {
    town.className = `town town-${name} phase-${step}`;
    if (scene[`${name}Alert`]) town.classList.add("is-alert");
    if (scene.complete) town.classList.add("is-complete");
  }
}

function schedule() {
  window.clearTimeout(timer);
  if (!playing || document.body.classList.contains("capture")) return;

  const delay = step === replay.length - 1 ? 2600 : 1450;
  timer = window.setTimeout(() => {
    render(step === replay.length - 1 ? 0 : step + 1);
    schedule();
  }, delay);
}

function setPlaying(value) {
  playing = value;
  const button = document.getElementById("play-toggle");
  button.textContent = playing ? "PAUSE" : "PLAY";
  button.setAttribute("aria-label", playing ? "Pause replay" : "Play replay");
  schedule();
}

document.getElementById("play-toggle").addEventListener("click", () => setPlaying(!playing));
document.getElementById("restart").addEventListener("click", () => {
  render(0);
  setPlaying(true);
});

const params = new URLSearchParams(window.location.search);
if (params.get("capture") === "1") {
  document.body.classList.add("capture");
  playing = false;
  render(Number.parseInt(params.get("step") || "0", 10));
} else {
  render(0);
  schedule();
}
