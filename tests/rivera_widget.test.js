// Tests for website/rivera.js (the Rivera widget): safe rendering, link safety, and which questions
// are answered in the browser versus sent to the server.
// No dependencies. Run from the project root:   node tests/rivera_widget.test.js
const assert = require('assert');
const path = require('path');
const R = require(path.join(__dirname, '..', 'website', 'rivera.js'));

const URLS = { dashboard: '/dashboard/', website: '/' };
let passed = 0;
function test(name, fn) { fn(); passed++; console.log('  ok  ', name); }
const nav = (q, page) => { const m = R.matchNav(q, page, URLS); return m ? (m.id || 'conversation') : null; };

console.log('rendering');
test('model output can never inject markup', () => {
  const html = R.renderRich('<img src=x onerror=alert(1)> and <script>alert(2)</script> & "quotes"');
  assert(!/<img|<script/i.test(html), html);
  assert(html.includes('&lt;img') && html.includes('&amp;'));
});
test('bold, bullets and paragraphs', () => {
  assert.strictEqual(R.renderRich('**Premium** recovers worst.'), '<p><strong>Premium</strong> recovers worst.</p>');
  assert.strictEqual(R.renderRich('Top:\n- one\n- two'), '<p>Top:</p><ul><li>one</li><li>two</li></ul>');
  assert.strictEqual(R.renderRich('a\n\nb'), '<p>a</p><p>b</p>');
});
test('bold markers cannot smuggle attributes', () => {
  assert(!/onclick/i.test(R.renderRich('**x" onclick="y**')) || R.renderRich('**x" onclick="y**').includes('&quot;'));
});

console.log('link safety');
test('only same-site paths, anchors and https links become buttons', () => {
  for (const ok of ['#/features', '#sec-upload', '/dashboard/', '../frontend/index.html', 'https://github.com/Harshavardhan810/Reverra'])
    assert.strictEqual(R.safeHref(ok), ok, ok);
  for (const bad of ['javascript:alert(1)', '//evil.example', 'http://insecure.example', 'data:text/html,hi', 'vbscript:x', ''])
    assert.strictEqual(R.safeHref(bad), '', bad);
});

console.log('website: navigation is instant, general and finance questions go to the server');
test('navigation questions', () => {
  assert.strictEqual(nav('What is Reverra?', 'website'), 'about');
  assert.strictEqual(nav('Show me the dashboard', 'website'), 'dashboard');
  assert.strictEqual(nav('How do I run it?', 'website'), 'run');
  assert.strictEqual(nav('where are the docs', 'website'), 'docs');
  assert.strictEqual(nav('docs', 'website'), 'docs');
  assert.strictEqual(nav('Is it open source?', 'website'), 'source');
  assert.strictEqual(nav('is it free', 'website'), 'pricing');
  assert.strictEqual(nav('what is it built with', 'website'), 'stack');
  assert.strictEqual(nav('is the data real?', 'website'), 'realdata');
  assert.strictEqual(nav('How do I contact you', 'website'), 'source');
  assert.strictEqual(nav('hi', 'website'), 'conversation');
  assert.strictEqual(nav('thanks!', 'website'), 'conversation');
});
test('the suggested website chips: three are navigation, the finance ones reach the model', () => {
  assert.strictEqual(nav('What is Reverra?', 'website'), 'about');
  assert.strictEqual(nav('Show me the dashboard', 'website'), 'dashboard');
  assert.strictEqual(nav('How do I run it?', 'website'), 'run');
  assert.strictEqual(nav('What is a good recovery rate?', 'website'), null);
  assert.strictEqual(nav('Explain NPV simply', 'website'), null);
});
test('a long, specific question about Reverra goes to the model, not the canned overview', () => {
  assert.strictEqual(nav('tell me something unusual about Reverra pricing strategies', 'website'), null);
  assert.strictEqual(nav('Tell me about Reverra', 'website'), 'about');
});
test('finance and general questions are NOT hijacked by site words', () => {
  for (const q of ['What is free cash flow?', 'What is a sales pipeline?', 'Explain the pricing strategy of a SaaS company',
                   'What is the theme of Hamlet?', 'What are the features of a bond?', 'How do I set up a budget?',
                   'What is a KPI dashboard?', 'What are the issues with DCF valuation?', 'Is real GDP data revised?',
                   'What is the cost of capital?', 'What is an open source license?', 'Calculate the EMI on 10 lakh at 8.5%',
                   'Who wrote Hamlet?', 'What is the capital of Australia?'])
    assert.strictEqual(nav(q, 'website'), null, q + ' -> ' + nav(q, 'website'));
});

console.log('dashboard: getting around, but data questions go to the server');
test('navigation questions', () => {
  assert.strictEqual(nav('How do I upload my data?', 'dashboard'), 'upload');
  assert.strictEqual(nav('upload csv', 'dashboard'), 'upload');
  assert.strictEqual(nav('where do I log a payment', 'dashboard'), 'log');
  assert.strictEqual(nav('pending', 'dashboard'), 'log');
  assert.strictEqual(nav('show me the at-risk customers', 'dashboard'), 'atrisk');
  assert.strictEqual(nav('where is the chart', 'dashboard'), 'chart');
  assert.strictEqual(nav('how do I simulate a payment', 'dashboard'), 'simulate');
  assert.strictEqual(nav('why is it read-only', 'dashboard'), 'lock');
  assert.strictEqual(nav('take me to the docs', 'dashboard'), 'website');
});
test('questions about the numbers are never treated as navigation', () => {
  for (const q of ['How many pending payments do we have?', 'Which failure reason loses the most money?', 'Which plan recovers worst?',
                   'Biggest unrecovered invoices', 'How accurate is the model?', 'Best day to retry?',
                   'Who are the top at-risk customers?', 'What is the recovery rate for premium customers?',
                   'what is a csv file', 'explain a pie chart'])
    assert.strictEqual(nav(q, 'dashboard'), null, q + ' -> ' + nav(q, 'dashboard'));
});
test('every dashboard chip goes to the server', () => {
  R.CHIPS.dashboard.forEach(c => assert.strictEqual(nav(c, 'dashboard'), null, c));
});

console.log('actions');
test('navigation answers carry working targets', () => {
  const m = R.matchNav('Show me the dashboard', 'website', URLS);
  assert.deepStrictEqual(m.actions.map(a => a.href), ['/dashboard/']);
  const up = R.matchNav('how do I upload data', 'dashboard', URLS);
  assert.deepStrictEqual(up.actions.map(a => a.href), ['#sec-upload']);
  const docs = R.matchNav('where are the docs', 'dashboard', { dashboard: '/dashboard/', website: '../website/index.html' });
  assert(docs.actions.some(a => a.href === '../website/index.html#/docs'));
  R.matchNav('What is Reverra?', 'website', URLS).actions.forEach(a => assert(R.safeHref(a.href), a.href));
});

console.log(`\n${passed} test groups passed`);
