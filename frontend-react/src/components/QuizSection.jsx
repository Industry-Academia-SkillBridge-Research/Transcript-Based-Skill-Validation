// src/components/QuizSection.jsx
import { useState } from "react";
import { prepareQuiz, submitQuiz } from "../api";

export function QuizSection({ studentId, onAfterQuiz }) {
  const [questions, setQuestions] = useState([]);
  const [result, setResult] = useState(null);
  const [status, setStatus] = useState("No quiz prepared yet.");

  async function handlePrepare() {
    const id = studentId.trim();
    if (!id) {
      alert("Enter a student ID first.");
      return;
    }
    try {
      setStatus("Preparing quiz...");
      const data = await prepareQuiz(id);
      setQuestions(data.questions || []);
      setResult(null);
      if (!data.questions || data.questions.length === 0) {
        setStatus("No questions returned.");
      } else {
        setStatus(`Loaded ${data.questions.length} questions.`);
      }
    } catch (err) {
      console.error(err);
      setStatus(err.message);
    }
  }

  async function handleSubmit() {
    const id = studentId.trim();
    if (!id) {
      alert("Enter a student ID first.");
      return;
    }
    if (!questions.length) {
      alert("Prepare a quiz first.");
      return;
    }
    const responses = [];
    questions.forEach((q) => {
      const qid = q.QuestionID ?? q.question_id;
      const radios = document.getElementsByName(`q_${qid}`);
      let chosen = null;
      for (const r of radios) {
        if (r.checked) {
          chosen = r.value;
          break;
        }
      }
      if (chosen) {
        responses.push({
          question_id: Number(qid),
          selected_option: chosen,
          response_time_seconds: 30,
        });
      }
    });
    if (!responses.length) {
      alert("Please answer at least one question.");
      return;
    }
    try {
      setStatus("Submitting quiz...");
      const data = await submitQuiz(id, responses);
      setResult(data);
      setStatus("Quiz scored.");
      if (onAfterQuiz) {
        onAfterQuiz(data);
      }
    } catch (err) {
      console.error(err);
      setStatus(err.message);
    }
  }

  function renderResultSummary() {
    if (!result) return <div className="muted">No results yet.</div>;

    const acc = (result.overall_accuracy * 100).toFixed(1);
    return (
      <div>
        <p>
          <strong>Answered:</strong> {result.num_answered} &nbsp;|&nbsp;
          <strong>Correct:</strong> {result.num_correct} &nbsp;|&nbsp;
          <strong>Overall accuracy:</strong> {acc}%
        </p>
        <p>
          <strong>Per-skill performance:</strong>
        </p>
        <ul>
          {result.per_skill && result.per_skill.length ? (
            result.per_skill.map((s) => {
              const pct = (s.accuracy * 100).toFixed(1);
              let tagClass = "";
              if (s.accuracy >= 0.7) tagClass = "good";
              else if (s.accuracy <= 0.3) tagClass = "bad";
              return (
                <li key={s.skill}>
                  <strong>{s.skill}</strong>: {s.num_correct}/{s.num_questions} correct{" "}
                  <span className={`tag ${tagClass}`}>{pct}%</span>
                </li>
              );
            })
          ) : (
            <li>No per-skill breakdown.</li>
          )}
        </ul>
      </div>
    );
  }

  return (
    <div className="card">
      <h2>Quiz</h2>
      <div className="quiz-header">
        <button className="secondary" onClick={handlePrepare}>
          Prepare Quiz
        </button>
        <button className="primary" onClick={handleSubmit}>
          Submit Quiz
        </button>
      </div>
      <div className="muted">{status}</div>

      <div className="quiz-questions">
        {questions.map((q, index) => {
          const qId = q.QuestionID ?? q.question_id ?? index;
          const options = {
            A: q.OptionA || q.optionA,
            B: q.OptionB || q.optionB,
            C: q.OptionC || q.optionC,
            D: q.OptionD || q.optionD,
          };
          return (
            <div key={qId} className="question">
              <div>
                <strong>Q{index + 1}.</strong>{" "}
                {q.QuestionText || q.question_text || "[no text]"}
                {q.Skill && (
                  <span className="tag" style={{ marginLeft: "0.5rem" }}>
                    {q.Skill}
                  </span>
                )}
              </div>
              <div>
                {["A", "B", "C", "D"].map((letter) => {
                  const text = options[letter];
                  if (!text) return null;
                  const name = `q_${qId}`;
                  const id = `q_${qId}_${letter}`;
                  return (
                    <div key={letter}>
                      <label>
                        <input type="radio" name={name} value={letter} id={id} />{" "}
                        <strong>{letter}</strong> – {text}
                      </label>
                    </div>
                  );
                })}
              </div>
            </div>
          );
        })}
        {!questions.length && (
          <div className="muted">No quiz prepared yet.</div>
        )}
      </div>

      <h3>Quiz result</h3>
      <div>{renderResultSummary()}</div>
    </div>
  );
}
