async function main() {
  const promptEl    = document.getElementById('prompt');
  const maxTokensEl = document.getElementById('maxTokens');
  const sendBtn     = document.getElementById('send');
  const messagesEl  = document.getElementById('messages');
  const emptyState  = document.getElementById('emptyState');
  const newChatBtn  = document.getElementById('newChat');

  function addRow(role, text, extraClass) {
    if (emptyState) emptyState.style.display = 'none';

    const row = document.createElement('div');
    row.className = 'message-row ' + role;

    const inner = document.createElement('div');
    inner.className = 'message-inner';

    const avatar = document.createElement('div');
    avatar.className = 'msg-avatar';
    avatar.textContent = role === 'user' ? 'You' : 'AI';

    const content = document.createElement('div');
    content.className = 'msg-content' + (extraClass || '');
    content.textContent = text;

    inner.appendChild(avatar);
    inner.appendChild(content);
    row.appendChild(inner);
    messagesEl.appendChild(row);
    messagesEl.scrollTop = messagesEl.scrollHeight;
    return content;
  }

  function autoResize() {
    promptEl.style.height = 'auto';
    promptEl.style.height = Math.min(promptEl.scrollHeight, 180) + 'px';
  }

  promptEl.addEventListener('input', autoResize);

  promptEl.addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendBtn.click();
    }
  });

  newChatBtn.addEventListener('click', () => {
    messagesEl.innerHTML = '';
    const empty = document.createElement('div');
    empty.className = 'empty-state';
    empty.id = 'emptyState';
    empty.innerHTML = '<div class="big-icon">🤖</div><h2>Mini LLM</h2><p>A small model running entirely on your CPU.</p>';
    messagesEl.appendChild(empty);
    promptEl.value = '';
    promptEl.style.height = 'auto';
    promptEl.focus();
  });

  async function sendPrompt() {
    const text = promptEl.value.trim();
    if (!text) return;
    const max_new_tokens = parseInt(maxTokensEl.value || '160', 10);

    addRow('user', text);
    promptEl.value = '';
    promptEl.style.height = 'auto';
    sendBtn.disabled = true;

    const botContent = addRow('bot', 'Thinking…', ' thinking');

    try {
      const resp = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt: text, max_new_tokens })
      });

      if (!resp.ok) {
        botContent.textContent = 'Error ' + resp.status;
        botContent.classList.remove('thinking');
        return;
      }

      const data = await resp.json();
      botContent.textContent = data.response || '(empty)';
      botContent.classList.remove('thinking');
    } catch (err) {
      botContent.textContent = 'Error: ' + String(err);
      botContent.classList.remove('thinking');
    } finally {
      sendBtn.disabled = false;
      promptEl.focus();
      messagesEl.scrollTop = messagesEl.scrollHeight;
    }
  }

  sendBtn.addEventListener('click', sendPrompt);
}

window.addEventListener('DOMContentLoaded', main);
