// ── Example conversations ────────────────────────────────────────────────────
const EXAMPLES = {
  baking: `Amanda: I baked chocolate chip cookies last night. They turned out really well!
Jerry: That sounds delicious! Do you want to share some?
Amanda: Of course! I'll bring a box to the office tomorrow.
Jerry: You're the best. I'll make coffee so we can enjoy them together.
Amanda: Perfect plan. See you at 9!`,

  project: `Sarah: Hey, have you finished the Q3 report yet?
Mike: Almost done. I just need to add the revenue charts and final recommendations.
Sarah: Great. The client meeting is on Friday so we need it by Thursday EOD.
Mike: No problem, I'll send a draft for your review by Wednesday afternoon.
Sarah: That works. Let me know if you need any data from the finance team.
Mike: Will do. I think the numbers look pretty strong this quarter.
Sarah: Fingers crossed! Last quarter was rough.`,

  travel: `Priya: I just booked our flights to Lisbon for next month!
Carlos: Amazing! How long are we staying?
Priya: Ten days. I was thinking we could spend the first three days in the city and then rent a car for the coast.
Carlos: The Algarve would be stunning. Can we stop in Sintra on the way?
Priya: Already on the list! I also found a great seafood restaurant near Alfama we should try.
Carlos: This is shaping up to be the best trip we've planned. Should I handle the hotel bookings?
Priya: That would be a huge help. Look for something central — maybe near Baixa.`,
};

// ── Example chip buttons ─────────────────────────────────────────────────────
document.getElementById("chip-baking").addEventListener("click", () => {
  document.getElementById("dialogue-input").value = EXAMPLES.baking;
});
document.getElementById("chip-project").addEventListener("click", () => {
  document.getElementById("dialogue-input").value = EXAMPLES.project;
});
document.getElementById("chip-travel").addEventListener("click", () => {
  document.getElementById("dialogue-input").value = EXAMPLES.travel;
});

// ── Avatar colour palette (cycles through speakers) ─────────────────────────
const AVATAR_COLORS = [
  "#fb923c", "#34d399", "#60a5fa", "#f472b6",
  "#a78bfa", "#fbbf24", "#f87171", "#2dd4bf",
];

function getAvatarColor(index) {
  return AVATAR_COLORS[index % AVATAR_COLORS.length];
}

// ── Helpers ──────────────────────────────────────────────────────────────────
function show(id) {
  document.getElementById(id).classList.remove("hidden");
}
function hide(id) {
  document.getElementById(id).classList.add("hidden");
}

// ── Form submission ──────────────────────────────────────────────────────────
document
  .getElementById("summarization-form")
  .addEventListener("submit", async (e) => {
    e.preventDefault();

    const dialogueInput = document.getElementById("dialogue-input");
    const summaryText   = document.getElementById("summary-text");
    const submitBtn     = document.getElementById("submit-btn");
    const personCards   = document.getElementById("person-cards");

    const dialogue = dialogueInput.value.trim();
    if (!dialogue) return;

    // Reset UI
    summaryText.innerText = "Processing…";
    submitBtn.disabled = true;
    submitBtn.textContent = "Summarizing…";
    personCards.innerHTML = "";
    show("summary-output");
    hide("per-person-output");

    try {
      const response = await fetch("/summarize/", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ dialogue }),
      });

      if (!response.ok) {
        const err = await response.json().catch(() => ({}));
        throw new Error(err.error || `Server error: ${response.status}`);
      }

      const data = await response.json();

      // Overall summary
      summaryText.innerText = data.summary || "No summary returned.";

      // Per-person summaries
      if (data.is_conversation && Array.isArray(data.per_person) && data.per_person.length) {
        show("per-person-output");
        data.per_person.forEach((person, idx) => {
          const color = getAvatarColor(idx);
          const initials = person.name
            .split(" ")
            .map((w) => w[0].toUpperCase())
            .join("")
            .slice(0, 2);

          const card = document.createElement("div");
          card.className = "person-card";
          card.innerHTML = `
            <div class="person-avatar" style="background:${color}">${initials}</div>
            <div class="person-body">
              <div class="person-name">${escapeHTML(person.name)}</div>
              <div class="person-summary">${escapeHTML(person.summary)}</div>
            </div>
          `;
          personCards.appendChild(card);
        });
      }
    } catch (err) {
      summaryText.innerText = `Error: ${err.message}`;
      hide("per-person-output");
    } finally {
      submitBtn.disabled = false;
      submitBtn.textContent = "Summarize";
    }
  });

// ── Utility ──────────────────────────────────────────────────────────────────
function escapeHTML(str) {
  return str
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}