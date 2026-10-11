/* AI Magic Vision frontend: polls /api/status, updates HUD, handles buttons. */
(function () {
  "use strict";

  var feed = document.getElementById("feed");
  var camDot = document.getElementById("camDot");
  var camText = document.getElementById("camText");
  var hudFps = document.getElementById("hudFps");
  var hudGesture = document.getElementById("hudGesture");
  var hudEffect = document.getElementById("hudEffect");
  var hudPerson = document.getElementById("hudPerson");
  var statusLine = document.getElementById("statusLine");
  var eventLog = document.getElementById("eventLog");
  var captureBar = document.getElementById("captureBar");
  var captureFill = document.getElementById("captureFill");
  var errorBanner = document.getElementById("errorBanner");
  var btnRecalibrate = document.getElementById("btnRecalibrate");
  var btnRestart = document.getElementById("btnRestart");

  var lastEffect = "";
  var pollFailures = 0;

  var GESTURE_LABELS = {
    open_palm: "\u270B Open Palm",
    fist: "\u270A Fist",
    point: "\uD83D\uDC49 Point",
    none: "--"
  };

  function now() {
    var d = new Date();
    function p(n) { return (n < 10 ? "0" : "") + n; }
    return p(d.getHours()) + ":" + p(d.getMinutes()) + ":" + p(d.getSeconds());
  }

  function logEvent(text) {
    var li = document.createElement("li");
    var t = document.createElement("span");
    t.className = "t";
    t.textContent = now();
    li.appendChild(t);
    li.appendChild(document.createTextNode(text));
    eventLog.insertBefore(li, eventLog.firstChild);
    while (eventLog.children.length > 8) {
      eventLog.removeChild(eventLog.lastChild);
    }
  }

  function setButtonsEnabled(enabled) {
    btnRecalibrate.disabled = !enabled;
    btnRestart.disabled = !enabled;
  }

  function poll() {
    fetch("/api/status")
      .then(function (r) {
        if (!r.ok) throw new Error("status " + r.status);
        return r.json();
      })
      .then(function (s) {
        pollFailures = 0;

        var online = !!s.camera_active;
        camDot.className = "dot " + (online ? "online" : "offline");
        camText.textContent = online ? "Camera Active" : "Camera Offline";
        setButtonsEnabled(online);

        syncFeatures(s);

        hudFps.textContent = s.fps != null ? s.fps : "--";
        var gLabel = GESTURE_LABELS[s.gesture] || s.gesture || "--";
        if ((!s.gesture || s.gesture === "none") &&
            s.finger_count >= 1 && s.finger_count <= 4) {
          gLabel = s.finger_count + " finger" + (s.finger_count > 1 ? "s" : "");
        }
        hudGesture.textContent = gLabel;
        hudEffect.textContent = s.effect_status || "--";
        hudPerson.textContent = s.person_detected
          ? (s.num_people > 1 ? s.num_people + " people!" : "Yes")
          : "No";

        statusLine.textContent = s.effect_status || "--";
        if (s.message) {
          statusLine.textContent += "  |  " + s.message;
        }

        if (s.effect_status && s.effect_status !== lastEffect) {
          logEvent(s.effect_status);
          lastEffect = s.effect_status;
        }

        if (s.capturing) {
          captureBar.classList.remove("hidden");
          captureFill.style.width = Math.round((s.capture_progress || 0) * 100) + "%";
        } else {
          captureBar.classList.add("hidden");
        }

        if (s.error && !online) {
          errorBanner.textContent = s.error;
          errorBanner.classList.remove("hidden");
        } else {
          errorBanner.classList.add("hidden");
        }
      })
      .catch(function () {
        pollFailures += 1;
        if (pollFailures === 3) {
          camDot.className = "dot offline";
          camText.textContent = "Backend unreachable";
          errorBanner.textContent =
            "Cannot reach the backend. Is the app still running?";
          errorBanner.classList.remove("hidden");
        }
      });
  }

  var FEATURE_LABELS = { eyes: "\uD83D\uDC41 Eye Rays", heart: "\u2764 Heart Reactor" };
  var featureBtns = document.querySelectorAll("[data-feature]");
  var btnBoot = document.getElementById("btnBoot");

  function syncFeatures(s) {
    if (!s.features) return;
    featureBtns.forEach(function (b) {
      var name = b.getAttribute("data-feature");
      var on = !!s.features[name];
      b.classList.toggle("on", on);
      b.textContent = FEATURE_LABELS[name] + ": " + (on ? "ON" : "OFF");
      b.title = s.features.face_tracking ? "" : "Face tracking unavailable (MediaPipe not loaded)";
    });
  }

  featureBtns.forEach(function (b) {
    b.addEventListener("click", function () {
      var name = b.getAttribute("data-feature");
      var turnOn = !b.classList.contains("on");
      fetch("/api/feature/" + name + "?on=" + turnOn, { method: "POST" })
        .then(function (r) { return r.json(); })
        .then(function (d) { logEvent(d.message || name); })
        .catch(function () { logEvent("Request failed: " + name); })
        .finally(poll);
    });
  });

  btnBoot.addEventListener("click", function () {
    fetch("/api/boot", { method: "POST" })
      .then(function (r) { return r.json(); })
      .then(function (d) { logEvent(d.message || "Boot sequence"); })
      .catch(function () { logEvent("Request failed: boot"); });
  });

  function postAction(url, label) {
    setButtonsEnabled(false);
    fetch(url, { method: "POST" })
      .then(function (r) { return r.json(); })
      .then(function (d) {
        logEvent(d.message || label);
        // Bust the MJPEG cache so the fresh stream restarts cleanly.
        feed.src = "/video_feed?ts=" + Date.now();
      })
      .catch(function () { logEvent("Request failed: " + label); })
      .finally(function () { setButtonsEnabled(true); poll(); });
  }

  btnRecalibrate.addEventListener("click", function () {
    if (confirm("Recalibrate background?\n\nPlease step OUT of the camera frame for 2 seconds.")) {
      postAction("/api/recalibrate", "Recalibrating background...");
    }
  });

  btnRestart.addEventListener("click", function () {
    postAction("/api/restart", "Experience restarted.");
  });

  // If the MJPEG stream errors (e.g. camera unplugged), retry periodically.
  feed.addEventListener("error", function () {
    setTimeout(function () { feed.src = "/video_feed?ts=" + Date.now(); }, 3000);
  });

  logEvent("Connecting to AI Magic Vision...");
  poll();
  setInterval(poll, 500);
})();
