"use strict";
const toggle = document.querySelector("#show-polygons");
if (toggle) toggle.addEventListener("change", () => {
  document.body.classList.toggle("hide-polygons", !toggle.checked);
});
const search = document.querySelector("#student-search");
const normalize = value => value.normalize("NFD").replace(/[\u0300-\u036f]/g, "").toLocaleLowerCase("cs");
if (search) search.addEventListener("input", () => {
  let visible = 0;
  const query = normalize(search.value.trim());
  document.querySelectorAll(".student-card").forEach(card => {
    card.hidden = !normalize(card.dataset.name).includes(query);
    if (!card.hidden) visible++;
  });
  document.querySelector("#empty-search").hidden = visible > 0;
});
const dialog = document.querySelector("#enlarged");
document.querySelectorAll(".enlarge").forEach(button => button.addEventListener("click", () => {
  const clone = button.closest(".panel").cloneNode(true);
  clone.querySelector(".enlarge").remove();
  dialog.querySelector(".dialog-content").replaceChildren(clone);
  dialog.showModal();
}));
dialog.querySelector(".close").addEventListener("click", () => dialog.close());
dialog.addEventListener("click", event => { if (event.target === dialog) {
  const rect = dialog.getBoundingClientRect();
  if (event.clientX < rect.left || event.clientX > rect.right || event.clientY < rect.top || event.clientY > rect.bottom) dialog.close();
}});
