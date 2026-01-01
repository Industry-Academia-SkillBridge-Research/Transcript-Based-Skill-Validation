/**
 * Quiz Security Controls
 * 
 * Implements practical security measures for quiz attempts:
 * - Fullscreen mode request
 * - Tab switch detection (visibility change)
 * - Copy/paste prevention
 * - Right-click prevention
 * - Violation logging
 */

let violations = [];
let isFullscreen = false;
let visibilityWarningShown = false;

/**
 * Initialize security controls for quiz
 */
export function initQuizSecurity(onViolation) {
  violations = [];
  visibilityWarningShown = false;
  
  // Request fullscreen
  requestFullscreen();
  
  // Setup event listeners
  setupVisibilityListener(onViolation);
  setupCopyPastePrevention(onViolation);
  setupRightClickPrevention(onViolation);
  setupKeyboardShortcuts(onViolation);
  
  // Cleanup function
  return () => {
    cleanupQuizSecurity();
  };
}

/**
 * Request fullscreen mode
 */
function requestFullscreen() {
  const element = document.documentElement;
  
  if (element.requestFullscreen) {
    element.requestFullscreen().then(() => {
      isFullscreen = true;
    }).catch((err) => {
      logViolation("fullscreen_denied", "User denied fullscreen request");
    });
  } else if (element.webkitRequestFullscreen) {
    element.webkitRequestFullscreen();
    isFullscreen = true;
  } else if (element.msRequestFullscreen) {
    element.msRequestFullscreen();
    isFullscreen = true;
  } else {
    logViolation("fullscreen_unavailable", "Fullscreen API not available");
  }
  
  // Listen for fullscreen changes
  document.addEventListener("fullscreenchange", handleFullscreenChange);
  document.addEventListener("webkitfullscreenchange", handleFullscreenChange);
  document.addEventListener("msfullscreenchange", handleFullscreenChange);
}

function handleFullscreenChange() {
  if (!document.fullscreenElement && 
      !document.webkitFullscreenElement && 
      !document.msFullscreenElement) {
    isFullscreen = false;
    logViolation("fullscreen_exit", "User exited fullscreen mode");
  }
}

/**
 * Setup visibility change detection (tab switch)
 */
function setupVisibilityListener(onViolation) {
  document.addEventListener("visibilitychange", () => {
    if (document.hidden) {
      if (!visibilityWarningShown) {
        visibilityWarningShown = true;
        const violation = {
          type: "tab_switch",
          timestamp: new Date().toISOString(),
          message: "User switched to another tab/window",
        };
        logViolation(violation.type, violation.message);
        if (onViolation) {
          onViolation(violation);
        }
        
        // Show warning when user returns
        window.addEventListener("focus", () => {
          alert("⚠️ Warning: Tab switch detected. Multiple violations may result in quiz disqualification.");
        }, { once: true });
      } else {
        logViolation("tab_switch_repeated", "User switched tabs multiple times");
      }
    }
  });
}

/**
 * Prevent copy/paste
 */
function setupCopyPastePrevention(onViolation) {
  // Prevent copy
  document.addEventListener("copy", (e) => {
    e.preventDefault();
    const violation = {
      type: "copy_attempt",
      timestamp: new Date().toISOString(),
      message: "User attempted to copy content",
    };
    logViolation(violation.type, violation.message);
    if (onViolation) {
      onViolation(violation);
    }
    return false;
  });
  
  // Prevent paste
  document.addEventListener("paste", (e) => {
    e.preventDefault();
    const violation = {
      type: "paste_attempt",
      timestamp: new Date().toISOString(),
      message: "User attempted to paste content",
    };
    logViolation(violation.type, violation.message);
    if (onViolation) {
      onViolation(violation);
    }
    return false;
  });
  
  // Prevent cut
  document.addEventListener("cut", (e) => {
    e.preventDefault();
    logViolation("cut_attempt", "User attempted to cut content");
    return false;
  });
  
  // Prevent context menu on text selection
  document.addEventListener("selectstart", (e) => {
    // Allow selection for radio buttons, but prevent text selection
    if (e.target.tagName !== "INPUT" && e.target.tagName !== "LABEL") {
      e.preventDefault();
      return false;
    }
  });
}

/**
 * Prevent right-click (context menu)
 */
function setupRightClickPrevention(onViolation) {
  document.addEventListener("contextmenu", (e) => {
    e.preventDefault();
    const violation = {
      type: "right_click",
      timestamp: new Date().toISOString(),
      message: "User attempted to open context menu",
    };
    logViolation(violation.type, violation.message);
    if (onViolation) {
      onViolation(violation);
    }
    return false;
  });
}

/**
 * Prevent keyboard shortcuts (Ctrl+C, Ctrl+V, F12, etc.)
 */
function setupKeyboardShortcuts(onViolation) {
  document.addEventListener("keydown", (e) => {
    // Prevent Ctrl+C, Ctrl+V, Ctrl+X, Ctrl+A
    if (e.ctrlKey && (e.key === "c" || e.key === "v" || e.key === "x" || e.key === "a")) {
      e.preventDefault();
      logViolation(`keyboard_shortcut_${e.key}`, `User attempted keyboard shortcut: Ctrl+${e.key.toUpperCase()}`);
      return false;
    }
    
    // Prevent F12 (Developer Tools)
    if (e.key === "F12") {
      e.preventDefault();
      logViolation("f12_pressed", "User attempted to open Developer Tools");
      return false;
    }
    
    // Prevent Ctrl+Shift+I (Developer Tools)
    if (e.ctrlKey && e.shiftKey && e.key === "I") {
      e.preventDefault();
      logViolation("devtools_shortcut", "User attempted to open Developer Tools");
      return false;
    }
    
    // Prevent Ctrl+Shift+J (Console)
    if (e.ctrlKey && e.shiftKey && e.key === "J") {
      e.preventDefault();
      logViolation("console_shortcut", "User attempted to open Console");
      return false;
    }
  });
}

/**
 * Log a violation
 */
function logViolation(type, message) {
  const violation = {
    type,
    message,
    timestamp: new Date().toISOString(),
    url: window.location.href,
    userAgent: navigator.userAgent,
  };
  violations.push(violation);
  console.warn(`[Quiz Security] ${type}: ${message}`);
}

/**
 * Get all logged violations
 */
export function getViolations() {
  return [...violations];
}

/**
 * Clear violations
 */
export function clearViolations() {
  violations = [];
}

/**
 * Cleanup security controls
 */
function cleanupQuizSecurity() {
  // Exit fullscreen if still active
  if (document.fullscreenElement || document.webkitFullscreenElement || document.msFullscreenElement) {
    if (document.exitFullscreen) {
      document.exitFullscreen();
    } else if (document.webkitExitFullscreen) {
      document.webkitExitFullscreen();
    } else if (document.msExitFullscreen) {
      document.msExitFullscreen();
    }
  }
  
  // Note: We don't remove all event listeners as they're scoped to the document
  // In a production app, you'd want to store references and remove them properly
}
