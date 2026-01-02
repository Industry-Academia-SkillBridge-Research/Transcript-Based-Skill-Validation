import { useState } from "react";

function QuizResultPage({ studentId, quizResult, questions, selectedSkills, onBack, onRetakeQuiz, onViewDashboard }) {
  const [reviewIndex, setReviewIndex] = useState(0);
  const [showReview, setShowReview] = useState(false);

  const currentReviewQuestion = showReview && quizResult ? questions[reviewIndex] : null;

  // Calculate per-skill performance
  const perSkillPerformance = {};
  if (quizResult?.per_question && questions) {
    questions.forEach((q) => {
      const skill = q.Skill || q.skill || "Unknown";
      if (!perSkillPerformance[skill]) {
        perSkillPerformance[skill] = {
          skill,
          total: 0,
          correct: 0,
        };
      }
      const result = quizResult.per_question.find(
        (item) => item.question_id === (q.QuestionID ?? q.question_id)
      );
      if (result) {
        perSkillPerformance[skill].total++;
        if (result.is_correct) {
          perSkillPerformance[skill].correct++;
        }
      }
    });
  }

  const skillPerformanceArray = Object.values(perSkillPerformance).map((item) => ({
    ...item,
    accuracy: item.total > 0 ? item.correct / item.total : 0,
  }));

  const overallAccuracy = quizResult?.accuracy ?? quizResult?.overall_accuracy ?? 0;
  const totalAnswered = quizResult?.total_answered ?? quizResult?.num_answered ?? 0;
  const correctAnswers = quizResult?.correct ?? quizResult?.num_correct ?? 0;

  const getAccuracyColor = (acc) => {
    if (acc >= 0.7) return "from-green-500 to-emerald-600";
    if (acc >= 0.4) return "from-yellow-500 to-orange-500";
    return "from-red-500 to-pink-600";
  };

  const getAccuracyTextColor = (acc) => {
    if (acc >= 0.7) return "text-green-600";
    if (acc >= 0.4) return "text-yellow-600";
    return "text-red-600";
  };

  return (
    <div className="max-w-4xl mx-auto space-y-6 min-h-screen pb-12">
      {/* Header */}
      <div className="flex items-center justify-between">
        <button
          onClick={onBack}
          className="flex items-center space-x-2 px-4 py-2 text-slate-600 hover:text-slate-900 hover:bg-white rounded-lg transition-colors"
        >
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
          </svg>
          <span>Back</span>
        </button>
        {showReview && (
          <div className="text-sm text-slate-600">
            Review {reviewIndex + 1} of {questions?.length || 0}
          </div>
        )}
      </div>

      {!showReview ? (
        <>
          {/* Title Section */}
          <div className="text-center mb-8">
            <div className="inline-flex items-center justify-center w-20 h-20 rounded-full bg-gradient-to-br from-green-500 to-emerald-600 mb-4 shadow-lg">
              <svg className="w-10 h-10 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
            </div>
            <h1 className="text-4xl font-bold text-slate-900 mb-2">Quiz Completed!</h1>
            <p className="text-slate-600 text-lg">Here's how you performed</p>
            {selectedSkills && selectedSkills.length > 0 && (
              <div className="mt-4 flex flex-wrap justify-center gap-2">
                {selectedSkills.map((skill, idx) => (
                  <span
                    key={idx}
                    className="px-3 py-1 bg-purple-100 text-purple-800 rounded-full text-sm font-medium"
                  >
                    {skill}
                  </span>
                ))}
              </div>
            )}
          </div>

          {/* Summary Cards */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
            <div className="bg-white rounded-xl p-6 border-2 border-green-200 shadow-md text-center">
              <div className="text-sm text-slate-600 mb-2 font-medium">Questions Answered</div>
              <div className="text-4xl font-bold text-green-600 mb-1">{totalAnswered}</div>
              <div className="text-xs text-slate-500">out of {questions?.length || 0}</div>
            </div>
            <div className="bg-white rounded-xl p-6 border-2 border-green-200 shadow-md text-center">
              <div className="text-sm text-slate-600 mb-2 font-medium">Correct Answers</div>
              <div className="text-4xl font-bold text-green-600 mb-1">{correctAnswers}</div>
              <div className="text-xs text-slate-500">answers</div>
            </div>
            <div className="bg-white rounded-xl p-6 border-2 border-green-200 shadow-md text-center">
              <div className="text-sm text-slate-600 mb-2 font-medium">Overall Accuracy</div>
              <div className={`text-4xl font-bold mb-1 ${getAccuracyTextColor(overallAccuracy)}`}>
                {(overallAccuracy * 100).toFixed(1)}%
              </div>
              <div className="text-xs text-slate-500">score</div>
            </div>
          </div>

          {/* Performance by Skill */}
          {skillPerformanceArray.length > 0 && (
            <div className="bg-white rounded-xl border border-slate-200 shadow-sm p-6 mb-8">
              <h3 className="text-xl font-bold text-slate-900 mb-4">Performance by Skill</h3>
              <div className="space-y-3">
                {skillPerformanceArray.map((skill, idx) => {
                  const accuracy = skill.accuracy * 100;
                  return (
                    <div key={idx} className="bg-slate-50 rounded-lg p-4 border border-slate-200">
                      <div className="flex items-center justify-between mb-2">
                        <span className="font-semibold text-slate-900">{skill.skill}</span>
                        <span className="text-sm text-slate-600">
                          {skill.correct}/{skill.total} correct
                        </span>
                      </div>
                      <div className="w-full bg-slate-200 rounded-full h-2 overflow-hidden">
                        <div
                          className={`h-full bg-gradient-to-r ${getAccuracyColor(skill.accuracy)} transition-all duration-500`}
                          style={{ width: `${accuracy}%` }}
                        />
                      </div>
                      <div className="text-right mt-1">
                        <span className={`text-sm font-semibold ${getAccuracyTextColor(skill.accuracy)}`}>
                          {accuracy.toFixed(1)}%
                        </span>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {/* Action Buttons */}
          <div className="flex flex-wrap justify-center gap-4">
            <button
              onClick={() => setShowReview(true)}
              className="px-8 py-3 bg-gradient-to-r from-blue-600 to-indigo-600 text-white rounded-lg font-semibold hover:from-blue-700 hover:to-indigo-700 transition-all shadow-lg hover:shadow-xl"
            >
              Review Answers with Explanations
            </button>
            {onViewDashboard && (
              <button
                onClick={onViewDashboard}
                className="px-8 py-3 bg-gradient-to-r from-indigo-600 to-purple-600 text-white rounded-lg font-semibold hover:from-indigo-700 hover:to-purple-700 transition-all shadow-lg hover:shadow-xl"
              >
                View Skill Profile Dashboard
              </button>
            )}
            {onRetakeQuiz && (
              <button
                onClick={onRetakeQuiz}
                className="px-8 py-3 bg-gradient-to-r from-purple-600 to-pink-600 text-white rounded-lg font-semibold hover:from-purple-700 hover:to-pink-700 transition-all shadow-lg hover:shadow-xl"
              >
                Retake Quiz
              </button>
            )}
          </div>
        </>
      ) : (
        <>
          {/* Review Mode */}
          <div className="bg-white rounded-xl border border-slate-200 shadow-sm p-4 mb-6">
            <div className="flex items-center justify-between">
              <h2 className="text-2xl font-bold text-slate-900">Review Answers</h2>
              <button
                onClick={() => setShowReview(false)}
                className="px-4 py-2 text-slate-600 hover:text-slate-900 hover:bg-slate-100 rounded-lg transition-colors"
              >
                Close Review
              </button>
            </div>
          </div>

          {currentReviewQuestion && (
            <div className="bg-white rounded-xl border border-slate-200 shadow-lg p-8">
              <div className="mb-6">
                <div className="flex items-center gap-3 mb-4">
                  <div className="flex items-center justify-center w-12 h-12 rounded-full bg-gradient-to-br from-blue-500 to-indigo-600 text-white font-bold text-xl shadow-md">
                    {reviewIndex + 1}
                  </div>
                  <div className="flex-1">
                    <h2 className="text-2xl font-bold text-slate-900 mb-2">
                      {currentReviewQuestion.QuestionText ?? currentReviewQuestion.question_text}
                    </h2>
                    {currentReviewQuestion.Skill && (
                      <span className="px-3 py-1 bg-blue-100 text-blue-800 rounded-full text-xs font-semibold">
                        {currentReviewQuestion.Skill}
                      </span>
                    )}
                  </div>
                </div>
              </div>

              <div className="space-y-3 mb-6">
                {["A", "B", "C", "D"].map((letter) => {
                  const optionText =
                    currentReviewQuestion[`Option${letter}`] ??
                    currentReviewQuestion[`option_${letter.toLowerCase()}`];
                  if (!optionText) return null;

                  const qId = currentReviewQuestion.QuestionID ?? currentReviewQuestion.question_id;
                  const resultItem = quizResult.per_question?.find(
                    (item) => item.question_id === qId
                  );
                  const isSelected = resultItem?.selected_option?.toUpperCase() === letter;
                  const isCorrect = resultItem?.correct_option?.toUpperCase() === letter;

                  return (
                    <div
                      key={letter}
                      className={`flex items-center gap-4 p-4 rounded-xl border-2 transition-all ${
                        isCorrect
                          ? "bg-green-50 border-green-500"
                          : isSelected
                          ? "bg-red-50 border-red-500"
                          : "bg-slate-50 border-slate-200"
                      }`}
                    >
                      <div
                        className={`flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center font-bold text-lg ${
                          isCorrect
                            ? "bg-green-500 text-white"
                            : isSelected
                            ? "bg-red-500 text-white"
                            : "bg-slate-300 text-slate-600"
                        }`}
                      >
                        {letter}
                      </div>
                      <span className="flex-1 text-slate-800 text-lg">{optionText}</span>
                      {isCorrect && (
                        <div className="flex items-center gap-2 text-green-700 font-semibold">
                          <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                          </svg>
                          Correct Answer
                        </div>
                      )}
                      {isSelected && !isCorrect && (
                        <div className="flex items-center gap-2 text-red-700 font-semibold">
                          <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                          </svg>
                          Your Answer
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>

              {/* Explanation Section */}
              <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 mb-6">
                <div className="flex items-start gap-3">
                  <svg
                    className="w-6 h-6 text-blue-600 flex-shrink-0 mt-0.5"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
                    />
                  </svg>
                  <div>
                    <h3 className="font-semibold text-blue-900 mb-1">Explanation</h3>
                    <p className="text-blue-800 text-sm">
                      {currentReviewQuestion.Explanation ??
                        currentReviewQuestion.explanation ??
                        "The correct answer is highlighted in green. Review the question and options to understand the concept better."}
                    </p>
                  </div>
                </div>
              </div>

              {/* Navigation */}
              <div className="flex items-center justify-between pt-6 border-t border-slate-200">
                <button
                  onClick={() => setReviewIndex(Math.max(0, reviewIndex - 1))}
                  disabled={reviewIndex === 0}
                  className="flex items-center gap-2 px-6 py-3 bg-slate-100 text-slate-700 rounded-lg font-semibold hover:bg-slate-200 transition-all disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
                  </svg>
                  Previous
                </button>

                <div className="text-sm text-slate-600">
                  Question {reviewIndex + 1} of {questions?.length || 0}
                </div>

                <button
                  onClick={() => setReviewIndex(Math.min((questions?.length || 0) - 1, reviewIndex + 1))}
                  disabled={reviewIndex === (questions?.length || 0) - 1}
                  className="flex items-center gap-2 px-6 py-3 bg-gradient-to-r from-blue-600 to-indigo-600 text-white rounded-lg font-semibold hover:from-blue-700 hover:to-indigo-700 transition-all shadow-lg hover:shadow-xl disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  Next
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                  </svg>
                </button>
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
}

export default QuizResultPage;

