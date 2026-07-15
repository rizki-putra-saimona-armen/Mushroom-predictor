// =====================================================================
// MycoLab — logika antarmuka
// 1. Membangun rosette (cetakan spora) sebagai SVG dinamis
// 2. Mengirim data form ke /predict lalu menganimasikan hasilnya
// =====================================================================

const SVG_NS = "http://www.w3.org/2000/svg";
const CENTER = 150;

const RINGS = [
  { radius: 34, count: 9 },
  { radius: 60, count: 15 },
  { radius: 86, count: 21 },
  { radius: 112, count: 27 },
  { radius: 138, count: 33 },
];

function buildRosette() {
  const svg = document.getElementById("rosette");

  // garis cincin tipis sebagai kerangka
  RINGS.forEach((ring) => {
    const circle = document.createElementNS(SVG_NS, "circle");
    circle.setAttribute("cx", CENTER);
    circle.setAttribute("cy", CENTER);
    circle.setAttribute("r", ring.radius);
    circle.classList.add("ring-line");
    svg.appendChild(circle);
  });

  // titik-titik spora tersebar di tiap cincin
  RINGS.forEach((ring, ringIndex) => {
    for (let i = 0; i < ring.count; i++) {
      const angle = (i / ring.count) * Math.PI * 2 + ringIndex * 0.15;
      const x = CENTER + ring.radius * Math.cos(angle);
      const y = CENTER + ring.radius * Math.sin(angle);

      const dot = document.createElementNS(SVG_NS, "circle");
      dot.setAttribute("cx", x.toFixed(2));
      dot.setAttribute("cy", y.toFixed(2));
      dot.setAttribute("r", 3.1);
      dot.classList.add("spore-dot");

      const delay = ringIndex * 55 + Math.round((i / ring.count) * 40);
      dot.style.setProperty("--delay", `${delay}ms`);

      svg.appendChild(dot);
    }
  });
}

function setRosetteState(state) {
  const svg = document.getElementById("rosette");
  svg.classList.remove("is-safe", "is-danger");
  // memaksa reflow supaya animasi warn-pulse bisa diputar ulang
  void svg.offsetWidth;
  if (state === "safe") svg.classList.add("is-safe");
  if (state === "danger") svg.classList.add("is-danger");
}

function labelFor(key) {
  const labels = {
    odor: "Bau", gill_color: "Warna Insang",
    ring_type: "Tipe Cincin", spore_print_color: "Warna Spora",
  };
  return labels[key] || key;
}

function renderResult(data) {
  const verdictWord = document.getElementById("verdict-word");
  const confidenceValue = document.getElementById("confidence-value");
  const detail = document.getElementById("result-detail");

  const isEdible = data.prediction === "edible";

  verdictWord.textContent = isEdible ? "Edible" : "Beracun";
  verdictWord.classList.remove("is-safe-text", "is-danger-text");
  verdictWord.classList.add(isEdible ? "is-safe-text" : "is-danger-text");
  confidenceValue.textContent = `keyakinan ${data.confidence}%`;

  setRosetteState(isEdible ? "safe" : "danger");

  const inputRows = Object.entries(data.input)
    .map(([k, v]) => `
      <div class="readout-row"><span>${labelFor(k)}</span><b>${v}</b></div>
    `).join("");

  detail.innerHTML = `
    <p class="verdict-headline ${isEdible ? "safe" : "danger"}">
      ${isEdible
        ? "Ciri-ciri ini konsisten dengan jamur yang aman dikonsumsi."
        : "Ciri-ciri ini konsisten dengan jamur beracun — jangan dikonsumsi."}
    </p>
    <div class="bars">
      <div class="bar-row">
        <span class="bar-label">Edible</span>
        <div class="bar-track"><div class="bar-fill safe" id="bar-edible"></div></div>
        <span class="bar-pct">${data.proba_edible}%</span>
      </div>
      <div class="bar-row">
        <span class="bar-label">Beracun</span>
        <div class="bar-track"><div class="bar-fill danger" id="bar-poison"></div></div>
        <span class="bar-pct">${data.proba_poisonous}%</span>
      </div>
    </div>
    <div class="readout">${inputRows}</div>
    <p class="disclaimer">
      Prediksi statistik berbasis data historis, bukan pengganti identifikasi
      oleh ahli mikologi. Jangan jadikan satu-satunya acuan sebelum
      mengonsumsi jamur liar.
    </p>
  `;

  requestAnimationFrame(() => {
    const barEdible = document.getElementById("bar-edible");
    const barPoison = document.getElementById("bar-poison");
    if (barEdible) barEdible.style.width = `${data.proba_edible}%`;
    if (barPoison) barPoison.style.width = `${data.proba_poisonous}%`;
  });
}

function setLoading(isLoading) {
  const btn = document.getElementById("analyze-btn");
  btn.disabled = isLoading;
  btn.querySelector("span").textContent = isLoading
    ? "Menganalisis…" : "Analisis Spesimen";
}

async function handleSubmit(event) {
  event.preventDefault();
  const form = event.target;
  const errorBox = document.getElementById("form-error");
  errorBox.textContent = "";

  const formData = new FormData(form);
  const payload = Object.fromEntries(formData.entries());

  const missing = Object.entries(payload).filter(([, v]) => !v);
  if (missing.length) {
    errorBox.textContent = "Lengkapi semua ciri-ciri sebelum menganalisis.";
    return;
  }

  setLoading(true);
  try {
    const res = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await res.json();

    if (!res.ok) {
      errorBox.textContent = data.error || "Terjadi kesalahan saat memprediksi.";
      return;
    }
    renderResult(data);
  } catch (err) {
    errorBox.textContent = "Tidak bisa terhubung ke server. Coba lagi.";
  } finally {
    setLoading(false);
  }
}

document.addEventListener("DOMContentLoaded", () => {
  buildRosette();
  document.getElementById("specimen-form").addEventListener("submit", handleSubmit);
});
