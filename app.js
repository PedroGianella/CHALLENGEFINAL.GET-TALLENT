const API_URL = "http://127.0.0.1:8000/ask";

const launcher = document.getElementById("launcher");
const chat = document.getElementById("chat");
const closeChat = document.getElementById("closeChat");

const messages = document.getElementById("messages");
const chatForm = document.getElementById("chatForm");
const questionInput = document.getElementById("questionInput");
const sendBtn = document.getElementById("sendBtn");

function addMessage(text, who) {
  const div = document.createElement("div");
  div.className = `msg ${who}`;
  div.textContent = text;
  messages.appendChild(div);
  messages.scrollTop = messages.scrollHeight;
}

function openChat() {
  chat.classList.remove("hidden");
  if (messages.childElementCount === 0) {
    addMessage("Hola. Consultame sobre Talleres. Respondo solo con el contexto cargado.", "bot");
  }
  setTimeout(() => questionInput.focus(), 0);
}

function closeChatFn() {
  chat.classList.add("hidden");
}

launcher.addEventListener("click", () => {
  if (chat.classList.contains("hidden")) openChat();
  else closeChatFn();
});

closeChat.addEventListener("click", closeChatFn);

chatForm.addEventListener("submit", async (e) => {
  e.preventDefault();

  const q = questionInput.value.trim();
  if (!q) return;

  addMessage(q, "user");
  questionInput.value = "";
  sendBtn.disabled = true;

  try {
    const resp = await fetch(API_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: q })
    });

    const data = await resp.json();

  
    const answer = (data && data.answer) ? data.answer : "No pude obtener respuesta.";
    addMessage(answer, "bot");
  } catch (err) {
    addMessage("Error al conectar con la API. Revisá que esté corriendo en http://127.0.0.1:8000.", "bot");
  } finally {
    sendBtn.disabled = false;
    questionInput.focus();
  }
});

