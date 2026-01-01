import { useState, useMemo, useEffect } from "react";
import { prepareQuiz, submitQuiz, getQuiz } from "../api";
import { shuffleArray, shuffleQuestionOptions } from "../utils/shuffle";

function QuizPage({ studentId, selectedSkills, quizId, timeLimitMinutes, sessionToken, sessionStartTime, onBack, onQuizCompleted }) {
  const [questions, setQuestions] = useState([]);
  const [answers, setAnswers] = useState({});
  const [quizResult, setQuizResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [quizGenerated, setQuizGenerated] = useState(false);
  const [currentQuestionIndex, setCurrentQuestionIndex] = useState(0);
  const [showReview, setShowReview] = useState(false);
  const [reviewIndex, setReviewIndex] = useState(0);
  
  // Store option mappings for each question (shuffled letter -> original letter)
  const [optionMappings, setOptionMappings] = useState({});

  // Load quiz if quizId is provided
  useEffect(() => {
    let isMounted = true;

    const loadQuizData = async () => {
      if (quizId) {
        try {
          setLoading(true);
          setError(null);
          const data = await getQuiz(quizId);
          
          // Only update state if component is still mounted
          if (isMounted) {
            setQuestions(data.questions || []);
            setQuizGenerated(true);
          }
        } catch (err) {
          if (isMounted) {
            setError(err.message || "Failed to load quiz");
            console.error("Failed to load quiz:", err);
          }
        } finally {
          if (isMounted) {
            setLoading(false);
          }
        }
      } else {
        // Reset if quizId is cleared
        if (isMounted) {
          setQuestions([]);
          setQuizGenerated(false);
          setAnswers({});
          setCurrentQuestionIndex(0);
        }
      }
    };

    loadQuizData();

    // Cleanup function
    return () => {
      isMounted = false;
    };
  }, [quizId]);
  
  // Shuffled questions and processed questions with shuffled options
  const processedQuestions = useMemo(() => {
    if (!questions.length) return [];
    
    // 1. Shuffle question order
    const shuffledQuestions = shuffleArray(questions);
    
    // 2. Shuffle options for each question and store mappings
    const mappings = {};
    const processed = shuffledQuestions.map((q) => {
      const qId = q.QuestionID ?? q.question_id;
      const { shuffledOptions, optionMapping } = shuffleQuestionOptions(q);
      mappings[qId] = optionMapping;
      
      // Create new question object with shuffled options
      const shuffledQuestion = { ...q };
      shuffledOptions.forEach((opt, index) => {
        const shuffledLetter = String.fromCharCode(65 + index); // A, B, C, D
        shuffledQuestion[`Option${shuffledLetter}`] = opt.text;
      });
      
      // Store original option letters for later mapping back
      shuffledQuestion._originalOptionMapping = optionMapping;
      
      return shuffledQuestion;
    });
    
    setOptionMappings(mappings);
    return processed;
  }, [questions]);

  const handleGenerateQuiz = async () => {
    if (!selectedSkills || selectedSkills.length === 0) {
      setError("Please select at least one skill.");
      return;
    }

    try {
      setLoading(true);
      setError(null);
      setQuizResult(null);
      setAnswers({});
      setCurrentQuestionIndex(0);
      setShowReview(false);

      const payload = {
        selected_skills: selectedSkills,
        num_questions_per_skill: 3,
        difficulty: "mixed",
      };

      const data = await prepareQuiz(studentId, payload);
      // Questions will be shuffled by useMemo when setQuestions is called
      setQuestions(data.questions || []);
      setQuizGenerated(true);
    } catch (err) {
      setError(err.message || "Failed to generate quiz");
    } finally {
      setLoading(false);
    }
  };

  const handleAnswerChange = (questionId, option) => {
    setAnswers((prev) => ({
      ...prev,
      [questionId]: option,
    }));
  };

  const handleNext = () => {
    if (currentQuestionIndex < processedQuestions.length - 1) {
      setCurrentQuestionIndex(currentQuestionIndex + 1);
    }
  };

  const handlePrevious = () => {
    if (currentQuestionIndex > 0) {
      setCurrentQuestionIndex(currentQuestionIndex - 1);
    }
  };

  const handleSubmitQuiz = async () => {
    if (processedQuestions.length === 0) {
      setError("No quiz questions available.");
      return;
    }

    // Map shuffled answers back to original option letters
    const responses = processedQuestions.map((q) => {
      const qId = q.QuestionID ?? q.question_id;
      const selectedShuffled = answers[qId]; // This is the shuffled letter (A/B/C/D)
      
      // Map back to original letter using option mapping
      const mapping = optionMappings[qId] || {};
      const originalOption = mapping[selectedShuffled] || selectedShuffled;
      
      return {
        question_id: Number(qId),
        selected_option: originalOption || "", // Send original A/B/C/D to backend
        response_time_seconds: 30,
      };
    });

    if (responses.filter((r) => r.selected_option).length === 0) {
      setError("Please answer at least one question.");
      return;
    }

    try {
      setLoading(true);
      setError(null);

      // Include quiz_id and session_token if available
      const data = await submitQuiz(studentId, responses, quizId || null, sessionToken || null);
      setQuizResult(data);

      if (onQuizCompleted) {
        // Pass original questions (not shuffled) for result page
        onQuizCompleted(data, questions);
      }
    } catch (err) {
      setError(err.message || "Failed to submit quiz");
    } finally {
      setLoading(false);
    }
  };

  // Get current question (from shuffled/processed questions)
  const currentQuestion = processedQuestions[currentQuestionIndex];
  const currentReviewQuestion = showReview && quizResult ? questions[reviewIndex] : null;

  // Calculate progress
  const progress = processedQuestions.length > 0 ? ((currentQuestionIndex + 1) / processedQuestions.length) * 100 : 0;
  const answeredCount = Object.keys(answers).length;

  // Check if all questions are answered
  const allAnswered = processedQuestions.length > 0 && Object.keys(answers).length === processedQuestions.length;

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
        {quizGenerated && !showReview && (
          <div className="text-sm text-slate-600">
            Question {currentQuestionIndex + 1} of {processedQuestions.length}
          </div>
        )}
      </div>

      {/* Title Section */}
      {!quizGenerated && (
        <div className="text-center mb-8">
          <div className="inline-flex items-center justify-center w-16 h-16 rounded-full bg-gradient-to-br from-purple-500 to-pink-600 mb-4 shadow-lg">
            <svg className="w-8 h-8 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
            </svg>
          </div>
          <h1 className="text-4xl font-bold text-slate-900 mb-2">Skill Validation Quiz</h1>
          <p className="text-slate-600 text-lg">
            Test your knowledge in the selected skills
          </p>
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
      )}

      {/* Error Message */}
      {error && (
        <div className="bg-red-50 border border-red-200 rounded-xl p-4 text-red-800">
          {error}
        </div>
      )}

      {/* Generate Quiz Button */}
      {!quizGenerated && (
        <div className="bg-white rounded-xl border border-slate-200 shadow-sm p-8 text-center">
          <p className="text-slate-600 mb-6">
            Click the button below to generate a quiz based on your selected skills.
          </p>
          <button
            onClick={handleGenerateQuiz}
            disabled={loading}
            className="px-8 py-3 bg-gradient-to-r from-purple-600 to-pink-600 text-white rounded-lg font-semibold hover:from-purple-700 hover:to-pink-700 transition-all shadow-lg hover:shadow-xl disabled:opacity-60 disabled:cursor-not-allowed"
          >
            {loading ? "Generating Quiz..." : "Generate Quiz"}
          </button>
        </div>
      )}

      {/* Quiz Progress Bar */}
      {quizGenerated && !showReview && questions.length > 0 && (
        <div className="bg-white rounded-xl border border-slate-200 shadow-sm p-4">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm font-medium text-slate-700">Progress</span>
            <span className="text-sm font-medium text-slate-700">
              {answeredCount} / {processedQuestions.length} answered
            </span>
          </div>
          <div className="w-full bg-slate-200 rounded-full h-3 overflow-hidden">
            <div
              className="h-full bg-gradient-to-r from-purple-500 to-pink-500 transition-all duration-300 ease-out rounded-full"
              style={{ width: `${progress}%` }}
            />
          </div>
        </div>
      )}

      {/* Single Question Display */}
      {quizGenerated && !showReview && currentQuestion && processedQuestions.length > 0 && (
        <div className="bg-white rounded-xl border border-slate-200 shadow-lg p-8">
          <div className="mb-6">
            <div className="flex items-center gap-3 mb-4">
              <div className="flex items-center justify-center w-12 h-12 rounded-full bg-gradient-to-br from-purple-500 to-pink-600 text-white font-bold text-xl shadow-md">
                {currentQuestionIndex + 1}
              </div>
              <div className="flex-1">
                <h2 className="text-2xl font-bold text-slate-900 mb-2">
                  {currentQuestion.QuestionText ?? currentQuestion.question_text}
                </h2>
                {currentQuestion.Skill && (
                  <div className="flex items-center gap-2">
                    <span className="px-3 py-1 bg-blue-100 text-blue-800 rounded-full text-xs font-semibold">
                      {currentQuestion.Skill}
                    </span>
                    {currentQuestion.Difficulty && (
                      <span className="px-3 py-1 bg-slate-100 text-slate-700 rounded-full text-xs font-medium">
                        {currentQuestion.Difficulty}
                      </span>
                    )}
                  </div>
                )}
              </div>
            </div>
          </div>

          <div className="space-y-3 mb-8">
            {["A", "B", "C", "D"].map((letter) => {
              const optionText = currentQuestion[`Option${letter}`] ?? currentQuestion[`option_${letter.toLowerCase()}`];
              if (!optionText) return null;
              
              const qId = currentQuestion.QuestionID ?? currentQuestion.question_id;
              const isSelected = answers[qId] === letter;

              return (
                <label
                  key={letter}
                  className={`flex items-center gap-4 p-4 rounded-xl cursor-pointer transition-all border-2 ${
                    isSelected
                      ? "bg-gradient-to-r from-purple-50 to-pink-50 border-purple-500 shadow-md"
                      : "bg-slate-50 border-slate-200 hover:bg-slate-100 hover:border-purple-300"
                  }`}
                >
                  <div className={`flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center font-bold text-lg ${
                    isSelected
                      ? "bg-gradient-to-br from-purple-500 to-pink-600 text-white"
                      : "bg-white border-2 border-slate-300 text-slate-600"
                  }`}>
                    {letter}
                  </div>
                  <input
                    type="radio"
                    name={`question_${qId}`}
                    value={letter}
                    checked={isSelected}
                    onChange={() => handleAnswerChange(qId, letter)}
                    className="w-5 h-5 text-purple-600 focus:ring-purple-500"
                  />
                  <span className="flex-1 text-slate-800 text-lg">{optionText}</span>
                  {isSelected && (
                    <svg className="w-6 h-6 text-purple-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                    </svg>
                  )}
                </label>
              );
            })}
          </div>

          {/* Navigation Buttons */}
          <div className="flex items-center justify-between pt-6 border-t border-slate-200">
            <button
              onClick={handlePrevious}
              disabled={currentQuestionIndex === 0}
              className="flex items-center gap-2 px-6 py-3 bg-slate-100 text-slate-700 rounded-lg font-semibold hover:bg-slate-200 transition-all disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
              </svg>
              Previous
            </button>

            <div className="flex gap-2">
              {processedQuestions.map((q, idx) => {
                const qId = q.QuestionID ?? q.question_id;
                const isAnswered = answers[qId];
                const isCurrent = idx === currentQuestionIndex;
                
                return (
                  <button
                    key={qId}
                    onClick={() => setCurrentQuestionIndex(idx)}
                    className={`w-10 h-10 rounded-lg font-semibold transition-all ${
                      isCurrent
                        ? "bg-gradient-to-br from-purple-500 to-pink-600 text-white shadow-md scale-110"
                        : isAnswered
                        ? "bg-green-100 text-green-700 hover:bg-green-200"
                        : "bg-slate-100 text-slate-600 hover:bg-slate-200"
                    }`}
                  >
                    {idx + 1}
                  </button>
                );
              })}
            </div>

            {currentQuestionIndex === processedQuestions.length - 1 ? (
              <button
                onClick={handleSubmitQuiz}
                disabled={loading || !allAnswered}
                className="flex items-center gap-2 px-6 py-3 bg-gradient-to-r from-green-600 to-emerald-600 text-white rounded-lg font-semibold hover:from-green-700 hover:to-emerald-700 transition-all shadow-lg hover:shadow-xl disabled:opacity-60 disabled:cursor-not-allowed"
              >
                {loading ? "Submitting..." : "Submit Quiz"}
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                </svg>
              </button>
            ) : (
              <button
                onClick={handleNext}
                className="flex items-center gap-2 px-6 py-3 bg-gradient-to-r from-purple-600 to-pink-600 text-white rounded-lg font-semibold hover:from-purple-700 hover:to-pink-700 transition-all shadow-lg hover:shadow-xl"
              >
                Next
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                </svg>
              </button>
            )}
          </div>
        </div>
      )}

      {/* Loading state after submission - navigating to result page */}
      {loading && quizResult && (
        <div className="bg-white rounded-xl border border-slate-200 shadow-sm p-8 text-center">
          <div className="inline-flex items-center justify-center w-16 h-16 rounded-full bg-gradient-to-br from-green-500 to-emerald-600 mb-4 shadow-lg">
            <svg className="w-8 h-8 text-white animate-spin" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
            </svg>
          </div>
          <p className="text-slate-600">Loading results...</p>
        </div>
      )}

      {/* Review Mode - removed as it's now in QuizResultPage */}
      {showReview && quizResult && currentReviewQuestion && (
        <div className="space-y-6">
          <div className="bg-white rounded-xl border border-slate-200 shadow-sm p-4">
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
                const optionText = currentReviewQuestion[`Option${letter}`] ?? currentReviewQuestion[`option_${letter.toLowerCase()}`];
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
                    <div className={`flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center font-bold text-lg ${
                      isCorrect
                        ? "bg-green-500 text-white"
                        : isSelected
                        ? "bg-red-500 text-white"
                        : "bg-slate-300 text-slate-600"
                    }`}>
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
                <svg className="w-6 h-6 text-blue-600 flex-shrink-0 mt-0.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
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
                Question {reviewIndex + 1} of {questions.length}
              </div>

              <button
                onClick={() => setReviewIndex(Math.min(questions.length - 1, reviewIndex + 1))}
                disabled={reviewIndex === questions.length - 1}
                className="flex items-center gap-2 px-6 py-3 bg-gradient-to-r from-blue-600 to-indigo-600 text-white rounded-lg font-semibold hover:from-blue-700 hover:to-indigo-700 transition-all shadow-lg hover:shadow-xl disabled:opacity-50 disabled:cursor-not-allowed"
              >
                Next
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                </svg>
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default QuizPage;
