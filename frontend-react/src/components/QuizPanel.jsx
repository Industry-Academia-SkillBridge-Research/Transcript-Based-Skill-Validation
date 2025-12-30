import { useState } from "react";

function QuizPanel({
  studentId,
  questions,
  setQuestions,
  quizResult,
  setQuizResult,
  apiBase,
  onQuizCompleted,
}) {
  const [loading, setLoading] = useState(false);
  const [localError, setLocalError] = useState(null);

  const handlePrepareQuiz = async () => {
    if (!studentId) {
      alert("Enter Student ID first.");
      return;
    }
    try {
      setLoading(true);
      setLocalError(null);
      setQuizResult(null);

      const res = await fetch(
        `${apiBase}/students/${encodeURIComponent(studentId)}/prepare-quiz`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({}),
        }
      );
      if (!res.ok) {
        const text = await res.text();
        throw new Error("HTTP " + res.status + ": " + text);
      }
      const data = await res.json();
      setQuestions(data.questions || []);
    } catch (err) {
      console.error(err);
      setLocalError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleSubmitQuiz = async () => {
    if (!studentId) {
      alert("Enter Student ID first.");
      return;
    }
    if (!questions.length) {
      alert("Prepare a quiz first.");
      return;
    }

    const responses = [];
    document.querySelectorAll("#quiz-questions .question").forEach((div) => {
      const qId = div.dataset.questionId;
      const checked = div.querySelector("input[type=radio]:checked");
      if (checked) {
        responses.push({
          question_id: Number(qId),
          selected_option: checked.value,
          response_time_seconds: 30, // placeholder
        });
      }
    });

    if (!responses.length) {
      alert("Please answer at least one question.");
      return;
    }

    try {
      setLoading(true);
      setLocalError(null);

      const res = await fetch(
        `${apiBase}/students/${encodeURIComponent(studentId)}/submit-quiz`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ responses }),
        }
      );
      if (!res.ok) {
        const text = await res.text();
        throw new Error("HTTP " + res.status + ": " + text);
      }
      const data = await res.json();
      setQuizResult(data);

      if (onQuizCompleted) {
        onQuizCompleted();
      }
    } catch (err) {
      console.error(err);
      setLocalError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const renderQuestions = () => {
    if (!questions.length) {
      return <p className="muted">No quiz prepared yet.</p>;
    }
    return (
      <div id="quiz-questions">
        {questions.map((q, idx) => {
          const qId = q.QuestionID ?? q.question_id ?? idx;
          const options = {
            A: q.OptionA,
            B: q.OptionB,
            C: q.OptionC,
            D: q.OptionD,
          };
          return (
            <div className="question" key={qId} data-question-id={qId}>
              <div>
                <strong>Q{idx + 1}.</strong>{" "}
                {q.QuestionText || q.question_text}
                {q.Skill && (
                  <span className="tag">
                    Skill: {q.Skill} ({q.Difficulty || "Unknown"})
                  </span>
                )}
              </div>
              <div style={{ marginTop: "0.25rem" }}>
                {["A", "B", "C", "D"].map((letter) => {
                  const text = options[letter];
                  if (!text) return null;
                  const name = `q_${qId}`;
                  const id = `q_${qId}_${letter}`;
                  return (
                    <div key={id}>
                      <label>
                        <input
                          type="radio"
                          name={name}
                          value={letter}
                          id={id}
                        />{" "}
                        <strong>{letter}</strong> – {text}
                      </label>
                    </div>
                  );
                })}
              </div>
            </div>
          );
        })}
      </div>
    );
  };

  const renderQuizResult = () => {
    if (!quizResult) {
      return <p className="muted">No quiz results yet.</p>;
    }

    const acc = (quizResult.overall_accuracy * 100).toFixed(1);
    const perSkill = (quizResult.per_skill || []).map((s, idx) => {
      const tagClass =
        s.accuracy >= 0.7 ? "good" : s.accuracy <= 0.3 ? "bad" : "";
      const pct = (s.accuracy * 100).toFixed(1);
      return (
        <li key={idx}>
          <strong>{s.skill}</strong>: {s.num_correct}/{s.num_questions} correct{" "}
          <span className={`tag ${tagClass}`}>{pct}%</span>
        </li>
      );
    });

    return (
      <div>
        <p>
          <strong>Answered:</strong> {quizResult.num_answered},{" "}
          <strong>Correct:</strong> {quizResult.num_correct},{" "}
          <strong>Overall accuracy:</strong> {acc}%
        </p>

        <p>
          <strong>Per-skill performance:</strong>
        </p>
        <ul>{perSkill.length ? perSkill : <li>No per-skill breakdown.</li>}</ul>

        <p style={{ fontSize: "0.8rem", color: "#6b7280" }}>
          {quizResult.note || ""}
        </p>
      </div>
    );
  };

  return (
    <section className="card">
      <h2>Step 3 · Quiz to validate your skills</h2>
      <p className="muted">
        The system generates questions for the weakest skills. Your answers help
        validate how strong you really are in those areas.
      </p>

      <button
        className="secondary"
        onClick={handlePrepareQuiz}
        disabled={loading || !studentId}
      >
        {loading ? "Working..." : "Prepare quiz"}
      </button>
      <button
        className="primary"
        onClick={handleSubmitQuiz}
        disabled={loading || !studentId}
      >
        Submit answers
      </button>

      {localError && (
        <p style={{ color: "#b91c1c", marginTop: "0.5rem" }}>{localError}</p>
      )}

      <div style={{ marginTop: "0.75rem" }}>{renderQuestions()}</div>

      <h3 style={{ marginTop: "1rem" }}>Quiz result</h3>
      {renderQuizResult()}
    </section>
  );
}

export default QuizPanel;
