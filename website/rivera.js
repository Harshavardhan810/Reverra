/*!
 * Rivera - the floating assistant for Reverra.
 *
 * One file, shared by the website and the dashboard (the dashboard loads it from the
 * website folder). It brings its own styles, so each page only needs:
 *
 *   <script>window.RIVERA_CONFIG = { page: 'website' };</script>     // or 'dashboard'
 *   <script src="rivera.js" defer></script>
 *
 * Questions about getting around the site or dashboard are answered right here in the
 * browser (instantly, even offline). Everything else goes to POST /ask on the backend.
 */
(function (root) {
  'use strict';

  /* ------------------------------------------------------------------ pure helpers */

  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, function (c) {
      return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c];
    });
  }

  // A tiny, safe markdown subset: **bold**, "- " bullet lists, blank-line paragraphs.
  // Everything is HTML-escaped first, so model output can never inject markup.
  function renderRich(text) {
    var lines = escapeHtml(text).split('\n');
    var out = [], list = null, para = [];
    function flushPara() { if (para.length) { out.push('<p>' + para.join('<br>') + '</p>'); para = []; } }
    function flushList() { if (list) { out.push('<ul>' + list.join('') + '</ul>'); list = null; } }
    lines.forEach(function (raw) {
      var line = raw.trim();
      var bullet = /^[-*\u2022]\s+(.*)$/.exec(line);
      if (bullet) { flushPara(); (list = list || []).push('<li>' + bullet[1] + '</li>'); }
      else if (!line) { flushPara(); flushList(); }
      else { flushList(); para.push(line); }
    });
    flushPara(); flushList();
    return out.join('').replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
  }

  // Only same-site paths, in-page anchors and https links may become buttons.
  function safeHref(h) {
    h = String(h || '');
    return /^(#|\/(?!\/)|\.\.?\/|https:\/\/)/.test(h) ? h : '';
  }

  function words(q) { return q.trim().split(/\s+/).filter(Boolean).length; }

  // "where / how do I / take me to ..." phrasing: the user is asking to be pointed somewhere.
  var NAV_PHRASE = /\b(where|how (do|can|would) (i|we)|take me|go to|show me|open|find|navigate|link|jump to|scroll|try|launch|visit)\b/i;

  /* ------------------------------------------------------------------ site knowledge */

  var REPO = 'https://github.com/Harshavardhan810/Reverra';

  function siteIntents(urls) {
    var DASH = { label: 'Open the dashboard', href: urls.dashboard };
    return [
      { id: 'about', loose: true, max: 7,
        re: /(what('?s| is| does)|tell me about|explain|describe|about)\b[^.?!]*\b(reverra|this (site|website|project|app|tool|product))\b|^\s*what is this\b/i,
        answer: '**Reverra** predicts which failed subscription payments are recoverable, drafts the recovery email, and tracks what actually came back, all in one pipeline. The Features page shows each step, and the dashboard lets you try it.',
        actions: [{ label: 'See the features', href: '#/features' }, DASH] },
      { id: 'features', loose: true,
        re: /\b(reverra|this|it|your|the site'?s?)\b[^.?!]{0,20}\bfeatures?\b|\bfeatures? (page|list)\b|\bwhat can (it|reverra|this) do\b|\bwhat does it do\b/i,
        answer: 'Reverra does four things: it **scores** each failed payment, **drafts** the recovery email, **answers questions** about the ledger (that is me), and **retrains** when you confirm real outcomes.',
        actions: [{ label: 'Open Features', href: '#/features' }] },
      { id: 'docs', loose: true,
        re: /\b(docs (page|section)|the docs|documentation|how (does|do) (it|reverra|this) work|how a payment moves|(reverra|this|the) (pipeline|architecture)|under the hood)\b|^\s*docs?\W*$/i,
        answer: 'The **Docs** page walks through how a payment moves from failure to confirmed outcome, the commands to run it locally, and the tech stack.',
        actions: [{ label: 'Open Docs', href: '#/docs' }] },
      { id: 'run', loose: true,
        re: /\b(get started|run (it|reverra|this|the (app|project|server|backend))( locally)?|run locally|install (it|reverra|this|the (app|project))|set ?up (it|reverra|this|the (app|project|server|backend))|clone (it|the repo)|how (do|can) i (run|start|install) (it|reverra|this))\b/i,
        answer: 'Running it locally takes four commands: install the requirements, generate the data, train the model, then start the server. They are listed under **Run it locally** on the Docs page.',
        actions: [{ label: 'Open Docs', href: '#/docs' }] },
      { id: 'dashboard', loose: false,
        re: /\b(dashboard|demo|live app)\b/i,
        answer: 'The **dashboard** is the live app: revenue by failure reason, a form to simulate a failed payment, pending payments, and an at-risk customer table.',
        actions: [DASH] },
      { id: 'stack', loose: true,
        re: /\b(tech(nology)? stack|(built|made) with|what (is|was) it (built|made) (with|in)|which (framework|language)s? (does|is|did) (it|reverra))\b/i,
        answer: 'Python, FastAPI and scikit-learn on the back end, Gemini for the emails and for me, and plain HTML, CSS and JavaScript on the front end. It is listed under **Built with** on the Docs page.',
        actions: [{ label: 'Open Docs', href: '#/docs' }] },
      { id: 'source', loose: true,
        re: /\b(is (it|reverra|this) open[- ]source|github|(the )?(source code|repo(sitory)?)|report (a )?(bug|issue)|file (a|an) issue|contact (page|you|the team|us)|get in touch|reach (you|out))\b/i,
        answer: 'Reverra is **open source**. Bugs and ideas go through GitHub issues, and the Contact page has the links.',
        actions: [{ label: 'Open Contact', href: '#/contact' }, { label: 'View the repo', href: REPO }] },
      { id: 'realdata', loose: true,
        re: /\b(is|are) (the|this|these|your)\b[^.?!]{0,25}\b(data|numbers?|customers?|payments?)\b[^.?!]{0,15}\b(real|fake|synthetic|simulated)\b|\breal (customer|payment) data\b|\bwhere does the data come from\b/i,
        answer: 'The data is **simulated**. No public dataset of real failed-payment outcomes exists, so the numbers are calibrated to published industry ranges. The pipeline, the time-based evaluation and the retraining loop are real, and the same columns accept real data.',
        actions: [] },
      { id: 'pricing', loose: true,
        re: /\bpricing page\b|\bhow much (does|is) (it|reverra)\b|\bis (it|reverra|this) (free|paid)\b|\bdoes (it|reverra) cost\b|\b(reverra|it|this)('s)? licen[sc]e\b|\bwhat licen[sc]e\b|\bcost to (use|run)\b/i,
        answer: 'Reverra is open source and free to run yourself, so there is no pricing page.',
        actions: [{ label: 'View the repo', href: REPO }] },
      { id: 'theme', loose: true,
        re: /\b(dark|light) (mode|theme)\b|\b(switch|change|toggle) (the )?theme\b/i,
        answer: 'Use the theme button in the top bar to switch between light and dark.', actions: [] },
      { id: 'home', loose: false,
        re: /^\s*(go )?(home|back (to )?(the )?(home|start|top))\b/i,
        answer: 'Taking you back to the start.', actions: [{ label: 'Go to Home', href: '#/' }] }
    ];
  }

  function dashboardIntents(urls) {
    var SITE = { label: 'Open the website', href: urls.website };
    return [
      { id: 'upload', loose: false,
        re: /\b(upload|csv|import)\b|\b(bring|use) (in )?my (own )?data\b|\breplace (the )?data\b/i,
        answer: '**Upload new data** takes a CSV with the required columns and retrains the model straight away. It sits just below the revenue chart.',
        actions: [{ label: 'Go to Upload', href: '#sec-upload' }] },
      { id: 'simulate', loose: false,
        re: /\b(simulate|simulation|what[- ]if|new entry|test a payment|recovery email|draft (an )?email|predict (a|one|my|this))\b/i,
        answer: 'The **New entry** form scores a hypothetical failed payment and drafts the recovery email. Fill in the plan, amount, tenure and failure reason, then run it.',
        actions: [{ label: 'Go to New entry', href: '#sec-simulate' }] },
      { id: 'log', loose: false,
        re: /\b(log|add|record|enter)\b[^.?!]{0,25}\b(payment|failure|failed)\b|\bconfirm(ation)?\b|\boutcomes?\b|\bpending\b/i,
        answer: '**Log a failed payment** adds it as pending. Once you confirm whether it was really recovered, it joins the training data and the model can retrain on it.',
        actions: [{ label: 'Go to Log a payment', href: '#sec-log' }, { label: 'See pending', href: '#sec-pending' }] },
      { id: 'atrisk', loose: false,
        re: /\bat[- ]risk\b|\bcustomers? table\b/i,
        answer: 'The **at-risk table** ranks unrecovered customers by how likely they are to come back, best bets first.',
        actions: [{ label: 'Go to At-risk customers', href: '#sec-atrisk' }] },
      { id: 'chart', loose: false,
        re: /\b(chart|graph|breakdown)\b/i,
        answer: '**Revenue by failure reason** shows recovered versus lost revenue for each reason, so you can see where money leaks.',
        actions: [{ label: 'Go to the chart', href: '#sec-reasons' }] },
      { id: 'regen', loose: false,
        re: /\b(regenerate|reset|new batch|fresh data)\b/i,
        answer: '**Regenerate data** in the top bar creates a fresh simulated batch and retrains the model. On the public demo it is locked to the site owner.',
        actions: [] },
      { id: 'lock', loose: true,
        re: /\b(read[- ]?only|unlock|admin token|locked)\b/i,
        answer: 'The public demo is **read-only**, so visitors cannot change the data. The site owner unlocks it with the admin token using the lock button in the top bar.',
        actions: [] },
      { id: 'theme', loose: true,
        re: /\b(dark|light) (mode|theme)\b|\b(switch|change|toggle) (the )?theme\b/i,
        answer: 'Use the sun/moon button in the top bar to switch between light and dark.', actions: [] },
      { id: 'website', loose: false,
        re: /\b(website|home ?page|landing|docs|documentation)\b/i,
        answer: 'The **Reverra website** has the overview, the features and the docs.',
        actions: [SITE, { label: 'Open the Docs', href: urls.website + '#/docs' }] },
      { id: 'realdata', loose: true,
        re: /\b(is|are) (the|this|these|your)\b[^.?!]{0,25}\b(data|numbers?|customers?|payments?)\b[^.?!]{0,15}\b(real|fake|synthetic|simulated)\b|\breal (customer|payment) data\b|\bwhere does the data come from\b/i,
        answer: 'The data is **simulated**: no public dataset of real failed-payment outcomes exists, so the numbers are calibrated to published industry ranges. The pipeline and retraining loop are real, and you can upload your own CSV with the same columns.',
        actions: [{ label: 'Go to Upload', href: '#sec-upload' }] }
    ];
  }

  var GREETING = /^\s*(hi+|hey+|hello+|yo|hola|namaste|good (morning|afternoon|evening)|sup)\b[\s!.?]*$/i;
  var THANKS = /^\s*(thanks?|thank you|thx|ty|cheers|great|awesome|cool|nice|perfect)\b[\s!.]*$/i;

  function matchNav(question, page, urls) {
    var q = question.trim();
    if (GREETING.test(q)) return { answer: page === 'website'
      ? 'Hi, I\'m **Rivera**. I can show you around Reverra, explain what it does, or help with finance questions. Where would you like to start?'
      : 'Hi, I\'m **Rivera**. Ask me about the numbers on this dashboard, how to use it, or a finance question.', actions: [] };
    if (THANKS.test(q)) return { answer: 'Anytime. Ask me anything else about the ledger or finance.', actions: [] };
    var list = page === 'website' ? siteIntents(urls) : dashboardIntents(urls);
    var navish = NAV_PHRASE.test(q) || words(q) <= 2;
    for (var i = 0; i < list.length; i++) {
      var it = list[i];
      if (!it.re.test(q)) continue;
      if (!it.loose && !navish) continue;
      if (it.max && words(q) > it.max) continue;
      return { answer: it.answer, actions: it.actions.slice(), id: it.id };
    }
    return null;
  }

  var CHIPS = {
    website: ['What is Reverra?', 'Show me the dashboard', 'How do I run it?', 'What is a good recovery rate?', 'Explain NPV simply'],
    dashboard: ['Which failure reason loses the most money?', 'Which plan recovers worst?', 'Biggest unrecovered invoices',
                'How accurate is the model?', 'Best day to retry?']
  };

  /* ------------------------------------------------------------------ styles */

  var CSS = [
    '.rv-root{--rv-font:inherit;position:fixed;right:max(28px,env(safe-area-inset-right));bottom:max(28px,env(safe-area-inset-bottom));z-index:9000;font-family:"Inter",system-ui,sans-serif;font-size:14px;line-height:1.5;color:var(--rv-text);}',
    '.rv-root *{box-sizing:border-box;}',
    /* tokens: the dashboard and the website name their colours differently, so map both */
    '.rv-dash{--rv-text:var(--ink);--rv-dim:var(--ink-muted);--rv-bg:var(--bg-a);--rv-btn:var(--green-deep);--rv-btn-text:var(--paper);--rv-strong:var(--green-deep);--rv-ok:var(--c-rec,#3DBE7A);--rv-bad:var(--c-lost,#E5534B);--rv-shadow:var(--shadow,0 10px 40px rgba(0,0,0,.3));}',
    '[data-theme="dark"] .rv-dash{--rv-btn:var(--green);--rv-strong:var(--green);}',
    '.rv-site{--rv-text:var(--fg);--rv-dim:var(--fg-dim);--rv-bg:var(--bg);--rv-btn:#E2592B;--rv-btn-text:#140502;--rv-strong:color-mix(in srgb,var(--vermilion,#D9481F) 80%,var(--fg));--rv-ok:#3DBE7A;--rv-bad:#E5534B;--rv-shadow:0 14px 48px rgba(0,0,0,.35);}',
    '.rv-root{--rv-edge:color-mix(in srgb,var(--rv-text) 15%,transparent);--rv-tint:color-mix(in srgb,var(--rv-text) 6%,transparent);--rv-surface:var(--rv-bg);}',
    '@supports (background:color-mix(in srgb,red 50%,blue)){.rv-root{--rv-surface:color-mix(in srgb,var(--rv-bg) 88%,transparent);}}',

    '.rv-launcher{display:flex;align-items:center;gap:12px;margin-left:auto;padding:8px 22px 8px 8px;border-radius:999px;cursor:pointer;color:var(--rv-text);background:var(--rv-surface);border:1px solid var(--rv-edge);box-shadow:var(--rv-shadow);-webkit-backdrop-filter:blur(18px) saturate(1.3);backdrop-filter:blur(18px) saturate(1.3);font:600 17px "Fraunces",Georgia,serif;letter-spacing:.01em;transition:transform .3s cubic-bezier(.2,.7,.2,1),border-color .25s,box-shadow .3s;}',
    '.rv-launcher:hover{transform:translateY(-3px);border-color:var(--rv-btn);}',
    '.rv-launcher:focus-visible,.rv-x:focus-visible,.rv-chip:focus-visible,.rv-act:focus-visible,.rv-send:focus-visible{outline:2px solid var(--rv-btn);outline-offset:2px;}',
    '.rv-av{flex:none;display:grid;place-items:center;width:38px;height:38px;border-radius:50%;background:var(--rv-btn);color:var(--rv-btn-text);font:italic 700 20px "Fraunces",Georgia,serif;}',

    '.rv-panel{position:absolute;right:0;bottom:calc(100% + 16px);width:390px;max-width:calc(100vw - 32px);max-height:min(700px,calc(100vh - 130px));max-height:min(700px,calc(100dvh - 130px));display:flex;flex-direction:column;border-radius:26px;background:var(--rv-surface);border:1px solid var(--rv-edge);box-shadow:var(--rv-shadow);-webkit-backdrop-filter:blur(24px) saturate(1.3);backdrop-filter:blur(24px) saturate(1.3);overflow:hidden;transform-origin:bottom right;opacity:0;visibility:hidden;transform:translateY(14px) scale(.97);transition:opacity .22s ease,transform .26s cubic-bezier(.2,.7,.2,1),visibility 0s linear .26s;}',
    '.rv-open .rv-panel{opacity:1;visibility:visible;transform:none;transition:opacity .22s ease,transform .26s cubic-bezier(.2,.7,.2,1),visibility 0s;}',
    '.rv-head{display:flex;align-items:center;gap:12px;padding:18px 22px 12px 22px;}',
    '.rv-title{flex:1;min-width:0;}',
    '.rv-title strong{display:block;font:700 20px/1.15 "Fraunces",Georgia,serif;}',
    '.rv-sub{display:flex;align-items:center;gap:7px;font-size:12.5px;color:var(--rv-dim);white-space:nowrap;}',
    '.rv-status{overflow:hidden;text-overflow:ellipsis;min-width:0;}',
    '.rv-dot{flex:none;width:7px;height:7px;border-radius:50%;background:var(--rv-ok);}',
    '.rv-off .rv-dot{background:var(--rv-bad);}',
    '.rv-x{flex:none;height:32px;border:0;border-radius:999px;background:transparent;color:var(--rv-dim);cursor:pointer;font:500 12.5px "Inter",sans-serif;padding:0 10px;transition:background .2s,color .2s;}',
    '.rv-clear{border:0;background:transparent;color:var(--rv-dim);font:inherit;text-decoration:underline;text-underline-offset:2px;cursor:pointer;padding:0 0 0 4px;}',
    '.rv-clear:hover,.rv-clear:focus-visible{color:var(--rv-text);}',
    '.rv-x{position:absolute;top:12px;right:12px;width:30px;height:30px;padding:0;font-size:22px;line-height:1;}',
    '.rv-x:hover{background:var(--rv-tint);color:var(--rv-text);}',

    '.rv-log{flex:1 1 auto;min-height:150px;overflow-y:auto;padding:6px 22px 8px;display:flex;flex-direction:column;gap:10px;scroll-behavior:smooth;overscroll-behavior:contain;}',
    '.rv-msg{max-width:88%;padding:11px 15px;border-radius:18px;animation:rv-in .28s cubic-bezier(.2,.7,.2,1) backwards;overflow-wrap:anywhere;}',
    '.rv-msg p{margin:0;} .rv-msg p+p{margin-top:7px;} .rv-msg ul{margin:6px 0 0;padding-left:18px;} .rv-msg li+li{margin-top:3px;}',
    '.rv-msg strong{color:var(--rv-strong);font-weight:700;}',
    '.rv-me{align-self:flex-end;background:var(--rv-btn);color:var(--rv-btn-text);border-bottom-right-radius:5px;}',
    '.rv-bot{align-self:flex-start;background:var(--rv-tint);border:1px solid var(--rv-edge);border-bottom-left-radius:5px;}',
    '.rv-bot.rv-err{border-color:var(--rv-bad);}',
    '.rv-acts{display:flex;flex-wrap:wrap;gap:7px;margin-top:10px;}',
    '.rv-act{display:inline-block;padding:6px 13px;border-radius:999px;border:1px solid var(--rv-btn);color:var(--rv-strong);font:600 12.5px "Inter",sans-serif;text-decoration:none;transition:background .2s,color .2s,transform .2s;}',
    '.rv-act:hover{background:var(--rv-btn);color:var(--rv-btn-text);transform:translateY(-1px);}',
    '.rv-typing{display:inline-flex;gap:5px;align-items:center;padding:4px 2px;}',
    '.rv-typing i{width:7px;height:7px;border-radius:50%;background:var(--rv-dim);animation:rv-dot 1s ease-in-out infinite;}',
    '.rv-typing i:nth-child(2){animation-delay:.15s;} .rv-typing i:nth-child(3){animation-delay:.3s;}',
    '.rv-slow{display:block;margin-top:6px;font-size:12px;color:var(--rv-dim);}',

    '.rv-chips{display:flex;flex-wrap:wrap;gap:8px;padding:6px 22px 10px;}',
    '.rv-chips:empty{display:none;}',
    '.rv-chip{padding:8px 15px;border-radius:999px;border:1px solid var(--rv-edge);background:transparent;color:var(--rv-text);font:400 13.5px "Inter",sans-serif;cursor:pointer;text-align:left;transition:border-color .2s,background .2s,transform .2s;}',
    '.rv-chip:hover{border-color:var(--rv-btn);background:var(--rv-tint);transform:translateY(-1px);}',

    '.rv-form{display:flex;align-items:flex-end;gap:8px;margin:4px 18px 8px;padding:7px 7px 7px 18px;border-radius:26px;border:1px solid var(--rv-edge);background:var(--rv-tint);transition:border-color .2s;}',
    '.rv-form:focus-within{border-color:var(--rv-btn);}',
    '.rv-in{flex:1;min-width:0;max-height:96px;resize:none;border:0;outline:none;background:transparent;color:var(--rv-text);font:400 14px/1.45 "Inter",sans-serif;padding:7px 0;}',
    '.rv-in::placeholder{color:var(--rv-dim);}',
    '.rv-send{flex:none;border:0;border-radius:999px;padding:10px 20px;background:var(--rv-btn);color:var(--rv-btn-text);font:700 14px "Fraunces",Georgia,serif;cursor:pointer;transition:transform .2s,opacity .2s;}',
    '.rv-send:hover{transform:translateY(-1px);} .rv-send:disabled{opacity:.5;cursor:default;transform:none;}',
    '.rv-note{margin:0 22px 14px;font-size:11px;color:var(--rv-dim);text-align:center;}',

    '.rv-flash{animation:rv-flash 1.6s ease;}',
    '@keyframes rv-in{from{opacity:0;transform:translateY(8px);}}',
    '@keyframes rv-dot{0%,80%,100%{opacity:.3;transform:translateY(0);}40%{opacity:1;transform:translateY(-3px);}}',
    '@keyframes rv-flash{0%,100%{box-shadow:0 0 0 0 transparent;}30%{box-shadow:0 0 0 6px color-mix(in srgb,var(--green,#D9481F) 35%,transparent);}}',
    '@media (max-width:520px){.rv-root{right:12px;left:12px;bottom:max(12px,env(safe-area-inset-bottom));}.rv-panel{width:auto;left:0;right:0;max-width:none;border-radius:22px;}.rv-launcher{margin-left:auto;}}',
    '@media (prefers-reduced-motion:reduce){.rv-root *{animation:none!important;transition:none!important;scroll-behavior:auto!important;}}',
    '@media print{.rv-root{display:none;}}'
  ].join('\n');

  // The host pages style bare `button` / `button:hover` globally. Prefixing every widget rule with
  // .rv-root raises its specificity so those page-wide rules can never leak into the widget.
  function scopeCss(css) {
    return css.replace(/(^|\})\s*([^@{}\s][^{}]*)\{/g, function (m, brace, sel) {
      var scoped = sel.split(',').map(function (part) {
        part = part.trim();
        if (/^(from|to|\d+%)$/.test(part)) return part;                              // keyframe steps
        if (/^(\.rv-root|\.rv-dash|\.rv-site|\[data-theme)/.test(part)) return part;  // token rules already target the root
        if (/^\.rv-(open|off)\b/.test(part)) return '.rv-root' + part;                // state classes sit on the root itself
        return '.rv-root ' + part;
      }).join(',');
      return brace + scoped + '{';
    });
  }

  /* ------------------------------------------------------------------ the widget */

  function init() {
    var CFG = root.RIVERA_CONFIG || {};
    var PAGE = CFG.page === 'dashboard' ? 'dashboard' : 'website';
    var isFile = location.protocol === 'file:';
    var urls = {
      dashboard: CFG.dashboardUrl || (isFile ? '../frontend/index.html' : '/dashboard/'),
      website: CFG.websiteUrl || (isFile ? '../website/index.html' : '/')
    };
    var API = CFG.api || ((isFile || (['localhost', '127.0.0.1'].indexOf(location.hostname) > -1 && location.port && location.port !== '8000'))
      ? 'http://localhost:8000' : location.origin);
    var STORE = 'rivera-chat-v1';
    var reduce = root.matchMedia && root.matchMedia('(prefers-reduced-motion: reduce)').matches;

    var style = document.createElement('style');
    style.id = 'rv-style'; style.textContent = scopeCss(CSS); document.head.appendChild(style);

    var host = document.createElement('div');
    // Plain divs on purpose: the host pages style bare <section>, <header> and <p> globally.
    host.className = 'rv-root ' + (PAGE === 'dashboard' ? 'rv-dash' : 'rv-site');
    host.innerHTML =
      '<div class="rv-panel" id="rv-panel" role="dialog" aria-label="Rivera, your assistant" aria-modal="false" inert>' +
        '<div class="rv-head"><span class="rv-av" aria-hidden="true">R</span>' +
          '<div class="rv-title"><strong>Rivera</strong><span class="rv-sub"><i class="rv-dot"></i><span class="rv-status"></span></span></div>' +
                    '<button type="button" class="rv-x" aria-label="Close Rivera">\u00d7</button></div>' +
        '<div class="rv-log" role="log" aria-live="polite" aria-label="Conversation with Rivera" tabindex="0"></div>' +
        '<div class="rv-chips" aria-label="Suggested questions"></div>' +
        '<form class="rv-form" autocomplete="off"><textarea class="rv-in" rows="1" maxlength="600" aria-label="Ask Rivera"></textarea>' +
          '<button type="submit" class="rv-send">Ask</button></form>' +
        '<div class="rv-note">Rivera can make mistakes. Not financial advice. <button type="button" class="rv-clear">Clear chat</button></div>' +
      '</div>' +
      '<button type="button" class="rv-launcher" aria-haspopup="dialog" aria-expanded="false" aria-controls="rv-panel">' +
        '<span class="rv-av" aria-hidden="true">R</span><span>Ask Rivera</span></button>';
    document.body.appendChild(host);

    var panel = host.querySelector('.rv-panel'), launcher = host.querySelector('.rv-launcher');
    var log = host.querySelector('.rv-log'), chips = host.querySelector('.rv-chips');
    var form = host.querySelector('.rv-form'), input = host.querySelector('.rv-in'), send = host.querySelector('.rv-send');
    var statusEl = host.querySelector('.rv-status');
    input.placeholder = PAGE === 'website' ? 'Ask Rivera about Reverra, finance, anything\u2026' : 'Ask Rivera anything about your payments\u2026';

    var state = { open: false, msgs: [], busy: false, asked: {}, llm: null };
    try { var saved = JSON.parse(sessionStorage.getItem(STORE) || 'null'); if (saved && Array.isArray(saved.msgs)) { state.msgs = saved.msgs.slice(-24); state.open = !!saved.open; state.asked = saved.asked || {}; } } catch (e) {}
    function persist() { try { sessionStorage.setItem(STORE, JSON.stringify({ msgs: state.msgs.slice(-24), open: state.open, asked: state.asked })); } catch (e) {} }

    function scrollDown() { log.scrollTop = log.scrollHeight; if (root.requestAnimationFrame) root.requestAnimationFrame(function () { log.scrollTop = log.scrollHeight; }); }

    function addActions(el, actions) {
      var acts = (actions || []).map(function (a) { return { label: a.label, href: safeHref(a.href) }; }).filter(function (a) { return a.href && a.label; });
      if (!acts.length) return;
      var wrap = document.createElement('div'); wrap.className = 'rv-acts';
      acts.forEach(function (a) {
        var link = document.createElement('a'); link.className = 'rv-act'; link.href = a.href; link.textContent = a.label + ' \u2192';
        if (/^https:/.test(a.href) && a.href.indexOf(location.origin) !== 0) { link.target = '_blank'; link.rel = 'noopener'; }
        wrap.appendChild(link);
      });
      el.appendChild(wrap);
    }

    function bubble(m) {
      var el = document.createElement('div');
      el.className = 'rv-msg ' + (m.role === 'user' ? 'rv-me' : 'rv-bot') + (m.error ? ' rv-err' : '');
      if (m.role === 'user') el.textContent = m.text; else { el.innerHTML = renderRich(m.text); addActions(el, m.actions); }
      return el;
    }

    function renderChips() {
      chips.textContent = '';
      if (state.busy) return;
      CHIPS[PAGE].filter(function (c) { return !state.asked[c]; }).slice(0, 4).forEach(function (c) {
        var b = document.createElement('button'); b.type = 'button'; b.className = 'rv-chip'; b.textContent = c;
        b.addEventListener('click', function () { ask(c); });
        chips.appendChild(b);
      });
      scrollDown();
    }

    function greet() {
      var g = matchNav('hi', PAGE, urls);
      state.msgs.push({ role: 'assistant', text: g.answer });
    }

    function renderAll() {
      log.textContent = '';
      state.msgs.forEach(function (m) { var el = bubble(m); el.style.animation = 'none'; log.appendChild(el); });
      renderChips();
    }

    function setOpen(open, focus) {
      state.open = open;
      host.classList.toggle('rv-open', open);
      launcher.setAttribute('aria-expanded', String(open));
      if (open) { panel.removeAttribute('inert'); if (!state.checked) checkHealth(); setTimeout(scrollDown, 30); if (focus !== false) setTimeout(function () { input.focus({ preventScroll: true }); }, 60); }
      else panel.setAttribute('inert', '');
      persist();
    }

    function setStatus(text, off) { statusEl.textContent = text; host.classList.toggle('rv-off', !!off); }
    function baseStatus() { return PAGE === 'website' ? 'Your guide to Reverra \u00b7 answers in seconds' : 'Reads your ledger \u00b7 answers in seconds'; }
    function checkHealth() {
      state.checked = true; setStatus(baseStatus());
      fetch(API + '/health').then(function (r) { return r.json(); }).then(function (d) {
        state.llm = !!d.llm_enabled;
        if (!d.llm_enabled) setStatus(d.read_only ? 'Instant answers on common questions' : 'Instant answers \u00b7 add GEMINI_API_KEY for more');
        else setStatus(baseStatus());
      }).catch(function () { state.checked = false; setStatus('Offline \u00b7 I can still help you get around', true); });
    }

    function follow(href) {                      // navigate for an action link
      if (href.charAt(0) === '#' && href.indexOf('#/') !== 0) {            // dashboard section anchor
        var t = document.getElementById(href.slice(1));
        if (t) { t.scrollIntoView({ behavior: reduce ? 'auto' : 'smooth', block: 'start' }); var card = t.querySelector('.glass') || t; card.classList.remove('rv-flash'); void card.offsetWidth; card.classList.add('rv-flash'); return true; }
      }
      if (href.indexOf('#/') === 0 && PAGE === 'website') { location.hash = href.slice(1); return true; }  // website route
      return false;                                                       // let the browser follow it
    }
    log.addEventListener('click', function (e) {
      var a = e.target.closest && e.target.closest('a.rv-act'); if (!a) return;
      var href = a.getAttribute('href');
      if (follow(href)) { e.preventDefault(); if (root.innerWidth <= 520) setOpen(false, false); }
    });

    function push(m) { state.msgs.push(m); var el = bubble(m); log.appendChild(el); scrollDown(); persist(); }

    function history() {
      return state.msgs.slice(0, -1).filter(function (m) { return !m.error; }).slice(-6)
        .map(function (m) { return { role: m.role === 'user' ? 'user' : 'assistant', text: String(m.text).slice(0, 700) }; });
    }

    function showTyping() {
      var el = document.createElement('div'); el.className = 'rv-msg rv-bot';
      el.innerHTML = '<span class="rv-typing" aria-label="Rivera is typing"><i></i><i></i><i></i></span>';
      log.appendChild(el); scrollDown(); return el;
    }

    function ask(question) {
      question = (question || '').replace(/\s+/g, ' ').trim();
      if (!question || state.busy) return;
      state.asked[question] = 1;
      push({ role: 'user', text: question });
      input.value = ''; autosize();

      var local = matchNav(question, PAGE, urls);               // instant, no network
      if (local) { push({ role: 'assistant', text: local.answer, actions: local.actions }); renderChips(); if (state.open) input.focus({ preventScroll: true }); return; }

      state.busy = true; send.disabled = true; renderChips();
      var typing = showTyping(), slowTimer = setTimeout(function () {
        var s = document.createElement('span'); s.className = 'rv-slow'; s.textContent = 'Still working on it\u2026'; typing.appendChild(s); scrollDown();
      }, 4500);
      var ctl = ('AbortController' in root) ? new AbortController() : null, kill = setTimeout(function () { if (ctl) ctl.abort(); }, 40000);

      fetch(API + '/ask', { method: 'POST', headers: { 'Content-Type': 'application/json' }, signal: ctl ? ctl.signal : undefined,
        body: JSON.stringify({ question: question, page: PAGE, history: history() }) })
        .then(function (r) { return r.json().catch(function () { return {}; }); })
        .then(function (d) {
          var failed = d.generated_by === 'error' || d.generated_by === 'rate_limited' || d.generated_by === 'unavailable';
          var text = d.answer || (d.detail && typeof d.detail === 'string' ? d.detail : 'I could not get an answer. Please try again.');
          if (Array.isArray(d.detail)) text = 'That question is not valid. Try shortening it.';
          return { role: 'assistant', text: text, actions: d.actions, error: failed };
        })
        .catch(function (e) {
          var timedOut = e && e.name === 'AbortError';
          setStatus('Offline \u00b7 I can still help you get around', true); state.checked = false;
          return { role: 'assistant', error: true, text: timedOut
            ? 'That is taking too long. Please try again in a moment.'
            : 'I cannot reach the server right now. Start the backend with: uvicorn backend.main:app --port 8000, then try again. I can still help you find your way around.' };
        })
        .then(function (m) {
          clearTimeout(slowTimer); clearTimeout(kill); if (typing.parentNode) typing.parentNode.removeChild(typing);
          push(m); state.busy = false; send.disabled = false; renderChips();
          if (state.open) input.focus({ preventScroll: true });
        });
    }

    function autosize() { input.style.height = 'auto'; input.style.height = Math.min(input.scrollHeight, 96) + 'px'; }
    input.addEventListener('input', autosize);
    input.addEventListener('keydown', function (e) { if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) { e.preventDefault(); ask(input.value); } });
    form.addEventListener('submit', function (e) { e.preventDefault(); ask(input.value); });
    launcher.addEventListener('click', function () { setOpen(!state.open); });
    host.querySelector('.rv-x').addEventListener('click', function () { setOpen(false, false); launcher.focus(); });
    host.querySelector('.rv-clear').addEventListener('click', function () { state.msgs = []; state.asked = {}; greet(); renderAll(); persist(); input.focus(); });
    document.addEventListener('keydown', function (e) { if (e.key === 'Escape' && state.open) { setOpen(false, false); launcher.focus(); } });
    launcher.addEventListener('pointerenter', function () { if (!state.checked) checkHealth(); }, { once: true });

    if (!state.msgs.length) greet();
    renderAll(); setStatus(baseStatus());
    if (state.open) setOpen(true, false);

    root.Rivera = { open: function () { setOpen(true); }, close: function () { setOpen(false, false); }, ask: function (q) { setOpen(true); ask(q); } };
  }

  /* ------------------------------------------------------------------ boot / exports */

  var api = { scopeCss: scopeCss, CSS: CSS, escapeHtml: escapeHtml, renderRich: renderRich, safeHref: safeHref, matchNav: matchNav, CHIPS: CHIPS };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  if (typeof document !== 'undefined') {
    if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init); else init();
  }
})(typeof window !== 'undefined' ? window : this);
