const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');

function setup() {
  const element = extra => ({events: {}, addEventListener(type, callback) {this.events[type] = callback;}, ...extra});
  const toggle = element({checked: true});
  const search = element({value: ''});
  const cards = [{dataset: {name: 'Eliška Černá'}}, {dataset: {name: 'Tomáš Bláha'}}];
  const empty = {};
  const classes = new Set();
  const close = element();
  const content = {replaceChildren(child) {this.child = child;}};
  const dialog = element({
    opened: false,
    showModal() {this.opened = true;}, close() {this.opened = false;},
    getBoundingClientRect() {return {left: 10, right: 100, top: 10, bottom: 100};},
    querySelector(selector) {return selector === '.close' ? close : content;}
  });
  let removed = false;
  const clone = {querySelector() {return {remove() {removed = true;}};}};
  const button = element({closest() {return {cloneNode() {return clone;}};}});
  const document = {
    body: {classList: {toggle(name, enabled) {if (enabled) classes.add(name); else classes.delete(name);}}},
    querySelector(selector) {return {'#show-polygons': toggle, '#student-search': search,
      '#empty-search': empty, '#enlarged': dialog}[selector];},
    querySelectorAll(selector) {return selector === '.student-card' ? cards : [button];}
  };
  vm.runInNewContext(fs.readFileSync(path.join(__dirname, '../scripts/classroom_viewer_assets/viewer.js'), 'utf8'), {document});
  return {toggle, search, cards, empty, classes, dialog, close, content, button, clone, removed: () => removed};
}

test('polygon checkbox toggles the shared page and dialog visibility class', () => {
  const s = setup();
  s.toggle.checked = false; s.toggle.events.change();
  assert.ok(s.classes.has('hide-polygons'));
  s.toggle.checked = true; s.toggle.events.change();
  assert.equal(s.classes.size, 0);
});

test('Czech student search ignores diacritics and reports empty results', () => {
  const s = setup();
  s.search.value = 'eliska'; s.search.events.input();
  assert.equal(s.cards[0].hidden, false); assert.equal(s.cards[1].hidden, true);
  assert.equal(s.empty.hidden, true);
  s.search.value = 'not a student'; s.search.events.input();
  assert.equal(s.empty.hidden, false);
  s.search.value = ''; s.search.events.input();
  assert.ok(s.cards.every(card => !card.hidden));
});

test('enlargement clones the selected panel and closes with button or backdrop', () => {
  const s = setup();
  s.button.events.click();
  assert.equal(s.dialog.opened, true); assert.equal(s.content.child, s.clone); assert.ok(s.removed());
  s.close.events.click(); assert.equal(s.dialog.opened, false);
  s.button.events.click();
  s.dialog.events.click({target: s.dialog, clientX: 50, clientY: 50});
  assert.equal(s.dialog.opened, true);
  s.dialog.events.click({target: s.dialog, clientX: 0, clientY: 0});
  assert.equal(s.dialog.opened, false);
});
