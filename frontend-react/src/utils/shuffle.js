/**
 * Fisher-Yates shuffle algorithm for fair randomization
 * @param {Array} array - Array to shuffle
 * @returns {Array} - New shuffled array
 */
export function shuffleArray(array) {
  const shuffled = [...array];
  for (let i = shuffled.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
  }
  return shuffled;
}

/**
 * Shuffle options for a question and return shuffled options + mapping
 * @param {Object} question - Question object with OptionA, OptionB, OptionC, OptionD
 * @returns {Object} - { shuffledOptions, optionMapping }
 *   - shuffledOptions: Array of {original: 'A'|'B'|'C'|'D', text: string}
 *   - optionMapping: Map from shuffled letter to original letter
 */
export function shuffleQuestionOptions(question) {
  const originalOptions = [
    { original: 'A', text: question.OptionA },
    { original: 'B', text: question.OptionB },
    { original: 'C', text: question.OptionC },
    { original: 'D', text: question.OptionD },
  ].filter(opt => opt.text && opt.text.trim()); // Filter out empty options

  const shuffledOptions = shuffleArray(originalOptions);
  
  // Create mapping: shuffled index -> original letter
  const optionMapping = {};
  shuffledOptions.forEach((opt, index) => {
    const shuffledLetter = String.fromCharCode(65 + index); // A, B, C, D
    optionMapping[shuffledLetter] = opt.original;
  });

  return {
    shuffledOptions,
    optionMapping, // Map from displayed letter to original letter
  };
}

