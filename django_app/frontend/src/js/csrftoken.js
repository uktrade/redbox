import { getCsrfToken } from "./utils";

document.body.addEventListener('submit', (evt) => {
  const form = evt.target;

  if (!form.querySelector('[name="csrfmiddlewaretoken"]')) {
    const value = getCsrfToken();

    if (value) {
      const input = document.createElement('input');
      input.type = 'hidden';
      input.name = 'csrfmiddlewaretoken';
      input.value = value;
      form.appendChild(input);
    }
  }
});

document.body.addEventListener('htmx:configRequest', (evt) => {
  const value = getCsrfToken();
  if (value) evt.detail.headers['X-CSRFToken'] = value;
});
