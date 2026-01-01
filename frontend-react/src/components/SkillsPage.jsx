import { useMemo, useState } from "react";

// Keywords to identify and merge similar technical skills
const TECHNICAL_KEYWORDS = {
  "SQL": ["sql", "database", "dbms", "relational database", "query", "schema"],
  "Python": ["python", "python programming"],
  "Java": ["java", "java programming", "java development"],
  "JavaScript": ["javascript", "js", "node.js", "nodejs"],
  "C++": ["c++", "cpp", "c plus plus"],
  "C": ["c programming", "c language"],
  "OOP": ["object-oriented", "oop", "object oriented programming", "classes", "inheritance", "polymorphism"],
  "Data Structures": ["data structures", "algorithms", "dsa"],
  "Machine Learning": ["machine learning", "ml", "neural network", "deep learning", "ai"],
  "Web Development": ["web development", "web", "html", "css", "frontend", "backend"],
  "Networking": ["networking", "network", "tcp/ip", "osi"],
  "Operating Systems": ["operating system", "os", "linux", "unix"],
  "Software Engineering": ["software engineering", "software development", "sdlc"],
  "Security": ["security", "cybersecurity", "cryptography"],
  "Statistics": ["statistics", "statistical", "probability"],
  "Data Visualization": ["data visualization", "visualization", "charts", "dashboards"],
};

// Categorize skills
function categorizeSkill(skillName) {
  const name = (skillName || "").toLowerCase();
  
  // Technical skills
  if (name.includes("programming") || name.includes("coding") || name.includes("development") ||
      name.includes("software") || name.includes("database") || name.includes("sql") ||
      name.includes("python") || name.includes("java") || name.includes("javascript") ||
      name.includes("web") || name.includes("network") || name.includes("security") ||
      name.includes("algorithm") || name.includes("data structure") || name.includes("oop") ||
      name.includes("machine learning") || name.includes("ai") || name.includes("operating system")) {
    return "Technical";
  }
  
  // Soft skills
  if (name.includes("communication") || name.includes("teamwork") || name.includes("leadership") ||
      name.includes("problem solving") || name.includes("critical thinking") || name.includes("analytical") ||
      name.includes("presentation") || name.includes("writing") || name.includes("professional") ||
      name.includes("ethics") || name.includes("workplace") || name.includes("soft skill")) {
    return "Soft Skills";
  }
  
  // Domain/Subject skills
  if (name.includes("mathematics") || name.includes("math") || name.includes("statistics") ||
      name.includes("business") || name.includes("management") || name.includes("project")) {
    return "Domain Knowledge";
  }
  
  return "Other";
}

// Extract keyword from skill name to merge similar skills
function extractKeyword(skillName) {
  const name = (skillName || "").toLowerCase();
  
  for (const [keyword, patterns] of Object.entries(TECHNICAL_KEYWORDS)) {
    for (const pattern of patterns) {
      if (name.includes(pattern)) {
        return keyword;
      }
    }
  }
  
  // If no keyword match, return first significant word
  const words = name.split(/\s+/).filter(w => w.length > 2);
  if (words.length > 0) {
    return words[0].charAt(0).toUpperCase() + words[0].slice(1);
  }
  
  return skillName;
}

// Merge similar skills
function mergeSimilarSkills(skills) {
  const merged = new Map();
  
  skills.forEach((skill) => {
    const skillName = skill.Skill || skill.skill || "";
    const keyword = extractKeyword(skillName);
    const score = Number(skill.ScoreNormalized ?? skill.FinalScore ?? skill.score ?? 0);
    const level = skill.FinalSkillLevel ?? skill.SkillLevel ?? skill.level ?? "Unknown";
    const evidence = skill.Evidence ?? skill.EvidenceCourses ?? skill.CourseEvidence ?? skill.MatchedCourses ?? "";
    
    if (merged.has(keyword)) {
      const existing = merged.get(keyword);
      // Take the maximum score and highest level
      const maxScore = Math.max(existing.score, score);
      existing.score = maxScore;
      existing.ScoreNormalized = maxScore;
      existing.FinalScore = maxScore;
      existing.originalNames.push(skillName);
      existing.evidence = existing.evidence ? `${existing.evidence}, ${evidence}` : evidence;
      // Update level if current is higher
      const levelOrder = { "Advanced": 3, "Intermediate": 2, "Developing": 2, "Beginner": 1, "Unknown": 0 };
      if (levelOrder[level] > levelOrder[existing.level]) {
        existing.level = level;
        existing.FinalSkillLevel = level;
        existing.SkillLevel = level;
      }
    } else {
      merged.set(keyword, {
        Skill: keyword,
        ScoreNormalized: score,
        FinalScore: score,
        score: score,
        FinalSkillLevel: level,
        SkillLevel: level,
        level: level,
        originalNames: [skillName],
        evidence: evidence,
        category: categorizeSkill(skillName),
      });
    }
  });
  
  return Array.from(merged.values());
}

function SkillLevelBadge({ level }) {
  const getLevelColor = (level) => {
    if (!level) return "bg-slate-100 text-slate-700";
    const l = level.toLowerCase();
    if (l.includes("advanced") || l.includes("expert")) return "bg-purple-100 text-purple-800 border-purple-300";
    if (l.includes("intermediate") || l.includes("proficient")) return "bg-blue-100 text-blue-800 border-blue-300";
    if (l.includes("developing") || l.includes("beginner")) return "bg-green-100 text-green-800 border-green-300";
    if (l.includes("beginner") || l.includes("novice")) return "bg-yellow-100 text-yellow-800 border-yellow-300";
    return "bg-slate-100 text-slate-700";
  };

  return (
    <span
      className={`inline-flex items-center px-3 py-1 rounded-full text-xs font-bold border ${getLevelColor(level)}`}
    >
      {level || "Unknown"}
    </span>
  );
}

function SkillCard({ skill, index }) {
  const score = Number(skill.ScoreNormalized ?? skill.FinalScore ?? skill.score ?? 0);
  const scorePercent = Math.min((score * 100), 100);
  const level = skill.FinalSkillLevel ?? skill.SkillLevel ?? skill.level ?? "Unknown";
  const evidence = skill.Evidence ?? skill.EvidenceCourses ?? skill.CourseEvidence ?? skill.MatchedCourses ?? "";

  // Determine progress bar color based on score
  const getProgressColor = (percent) => {
    if (percent >= 75) return "from-green-500 to-emerald-600";
    if (percent >= 50) return "from-blue-500 to-indigo-600";
    if (percent >= 25) return "from-yellow-500 to-orange-500";
    return "from-red-400 to-red-500";
  };

  return (
    <div className="bg-white rounded-xl border border-slate-200 shadow-sm hover:shadow-md transition-all p-6">
      <div className="flex items-start justify-between mb-4">
        <div className="flex-1">
          <div className="flex items-center gap-3 mb-2">
            <div className="flex items-center justify-center w-10 h-10 rounded-lg bg-gradient-to-br from-blue-500 to-indigo-600 text-white font-bold text-lg shadow-md">
              {index + 1}
            </div>
            <h3 className="text-lg font-bold text-slate-900">{skill.Skill || skill.skill || "Unknown Skill"}</h3>
          </div>
          <div className="flex items-center gap-3 mt-3">
            <SkillLevelBadge level={level} />
            <span className="text-sm font-semibold text-slate-600">
              Score: <span className="text-blue-600">{scorePercent.toFixed(1)}%</span>
            </span>
          </div>
        </div>
      </div>

      {/* Progress Bar */}
      <div className="mb-4">
        <div className="flex items-center justify-between text-xs text-slate-600 mb-2">
          <span>Skill Proficiency</span>
          <span className="font-semibold">{scorePercent.toFixed(1)}%</span>
        </div>
        <div className="w-full bg-slate-200 rounded-full h-3 overflow-hidden">
          <div
            className={`h-full bg-gradient-to-r ${getProgressColor(scorePercent)} transition-all duration-500 ease-out rounded-full`}
            style={{ width: `${scorePercent}%` }}
          />
        </div>
      </div>

      {/* Original Skill Names (if merged) */}
      {skill.originalNames && skill.originalNames.length > 1 && (
        <div className="mt-3 pt-3 border-t border-slate-100">
          <div className="text-xs font-semibold text-slate-500 uppercase tracking-wide mb-2">
            Combined From
          </div>
          <div className="flex flex-wrap gap-2">
            {skill.originalNames.map((name, idx) => (
              <span key={idx} className="text-xs px-2 py-1 bg-slate-100 text-slate-600 rounded">
                {name}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Evidence from Modules */}
      {evidence && (
        <div className="mt-4 pt-4 border-t border-slate-100">
          <div className="text-xs font-semibold text-slate-500 uppercase tracking-wide mb-2">
            Derived From Modules
          </div>
          <div className="text-sm text-slate-700 bg-slate-50 rounded-lg p-3">
            {typeof evidence === 'string' ? evidence : JSON.stringify(evidence)}
          </div>
        </div>
      )}
    </div>
  );
}

export default function SkillsPage({ skills, studentName, studentId, onContinue, onBack }) {
  const [activeCategory, setActiveCategory] = useState("All");

  // Merge similar skills and categorize
  const processedSkills = useMemo(() => {
    if (!skills?.length) return [];
    return mergeSimilarSkills(skills);
  }, [skills]);

  // Categorize skills
  const categorizedSkills = useMemo(() => {
    const categories = {
      "All": processedSkills,
      "Technical": processedSkills.filter(s => s.category === "Technical"),
      "Soft Skills": processedSkills.filter(s => s.category === "Soft Skills"),
      "Domain Knowledge": processedSkills.filter(s => s.category === "Domain Knowledge"),
      "Other": processedSkills.filter(s => s.category === "Other"),
    };
    return categories;
  }, [processedSkills]);

  // Sort skills by score
  const sortedSkills = useMemo(() => {
    const categorySkills = categorizedSkills[activeCategory] || [];
    return [...categorySkills].sort((a, b) => {
      const scoreA = Number(a.ScoreNormalized ?? a.FinalScore ?? a.score ?? 0);
      const scoreB = Number(b.ScoreNormalized ?? b.FinalScore ?? b.score ?? 0);
      return scoreB - scoreA;
    });
  }, [categorizedSkills, activeCategory]);

  // Get category counts
  const categoryCounts = useMemo(() => {
    return {
      All: categorizedSkills.All.length,
      Technical: categorizedSkills.Technical.length,
      "Soft Skills": categorizedSkills["Soft Skills"].length,
      "Domain Knowledge": categorizedSkills["Domain Knowledge"].length,
      Other: categorizedSkills.Other.length,
    };
  }, [categorizedSkills]);

  // Calculate statistics
  const stats = useMemo(() => {
    const allSkills = categorizedSkills.All || [];
    if (!allSkills.length) {
      return { total: 0, advanced: 0, intermediate: 0, beginner: 0, avgScore: 0 };
    }

    let advanced = 0;
    let intermediate = 0;
    let beginner = 0;
    let totalScore = 0;

    allSkills.forEach((skill) => {
      const level = (skill.FinalSkillLevel ?? skill.SkillLevel ?? skill.level ?? "").toLowerCase();
      const score = Number(skill.ScoreNormalized ?? skill.FinalScore ?? skill.score ?? 0);
      totalScore += score;

      if (level.includes("advanced") || level.includes("expert")) {
        advanced++;
      } else if (level.includes("intermediate") || level.includes("proficient") || level.includes("developing")) {
        intermediate++;
      } else {
        beginner++;
      }
    });

    return {
      total: allSkills.length,
      advanced,
      intermediate,
      beginner,
      avgScore: (totalScore / allSkills.length) * 100,
    };
  }, [categorizedSkills]);

  return (
    <div className="max-w-6xl mx-auto space-y-6">
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
        <button
          onClick={onContinue}
          className="px-6 py-2.5 bg-gradient-to-r from-blue-600 to-indigo-600 text-white rounded-lg font-semibold hover:from-blue-700 hover:to-indigo-700 transition-all shadow-lg hover:shadow-xl"
        >
          Continue to Dashboard
        </button>
      </div>

      {/* Title Section */}
      <div className="text-center mb-8">
        <div className="inline-flex items-center justify-center w-16 h-16 rounded-full bg-gradient-to-br from-indigo-500 to-purple-600 mb-4 shadow-lg">
          <svg className="w-8 h-8 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
          </svg>
        </div>
        <h1 className="text-4xl font-bold text-slate-900 mb-2">Your Skill Profile</h1>
        <p className="text-slate-600 text-lg">
          Skills identified from your academic modules and performance
        </p>
        {studentName && (
          <p className="text-slate-500 mt-1">for {studentName}</p>
        )}
      </div>

      {/* Statistics Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="bg-gradient-to-br from-blue-50 to-indigo-50 rounded-xl p-6 border border-blue-200">
          <div className="text-sm text-slate-600 mb-2">Total Skills</div>
          <div className="text-3xl font-bold text-blue-600">{stats.total}</div>
        </div>
        <div className="bg-gradient-to-br from-purple-50 to-pink-50 rounded-xl p-6 border border-purple-200">
          <div className="text-sm text-slate-600 mb-2">Advanced</div>
          <div className="text-3xl font-bold text-purple-600">{stats.advanced}</div>
        </div>
        <div className="bg-gradient-to-br from-blue-50 to-cyan-50 rounded-xl p-6 border border-blue-200">
          <div className="text-sm text-slate-600 mb-2">Intermediate</div>
          <div className="text-3xl font-bold text-blue-600">{stats.intermediate}</div>
        </div>
        <div className="bg-gradient-to-br from-green-50 to-emerald-50 rounded-xl p-6 border border-green-200">
          <div className="text-sm text-slate-600 mb-2">Average Score</div>
          <div className="text-3xl font-bold text-green-600">{stats.avgScore.toFixed(1)}%</div>
        </div>
      </div>

      {/* Category Tabs */}
      <div className="bg-white rounded-xl border border-slate-200 shadow-sm p-2">
        <div className="flex flex-wrap gap-2">
          {["All", "Technical", "Soft Skills", "Domain Knowledge", "Other"].map((category) => (
            <button
              key={category}
              onClick={() => setActiveCategory(category)}
              className={`px-4 py-2 rounded-lg font-medium transition-all ${
                activeCategory === category
                  ? "bg-gradient-to-r from-blue-600 to-indigo-600 text-white shadow-md"
                  : "bg-slate-100 text-slate-700 hover:bg-slate-200"
              }`}
            >
              {category}
              <span className="ml-2 text-xs opacity-75">({categoryCounts[category] || 0})</span>
            </button>
          ))}
        </div>
      </div>

      {/* Skills Grid */}
      <div>
        <div className="mb-6">
          <h2 className="text-2xl font-bold text-slate-900 mb-2">
            {activeCategory === "All" ? "All Skills" : `${activeCategory} Skills`}
          </h2>
          <p className="text-slate-600">
            {activeCategory === "All" 
              ? "Your skill levels are calculated based on module grades and course completion"
              : `Showing ${sortedSkills.length} ${activeCategory.toLowerCase()} skill${sortedSkills.length !== 1 ? 's' : ''}`
            }
          </p>
        </div>

        {!sortedSkills.length ? (
          <div className="bg-white rounded-xl border border-slate-200 p-12 text-center">
            <svg className="w-16 h-16 text-slate-400 mx-auto mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
            </svg>
            <p className="text-slate-600 text-lg">No skills found</p>
            <p className="text-slate-500 text-sm mt-2">Skills will be displayed here once processed</p>
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {sortedSkills.map((skill, index) => (
              <SkillCard key={index} skill={skill} index={index} />
            ))}
          </div>
        )}
      </div>

      {/* Info Box */}
      <div className="bg-gradient-to-r from-blue-50 to-indigo-50 rounded-xl p-6 border border-blue-200">
        <div className="flex items-start gap-4">
          <div className="flex-shrink-0">
            <svg className="w-6 h-6 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
          </div>
          <div>
            <h3 className="font-semibold text-slate-900 mb-1">How are skill levels calculated?</h3>
            <p className="text-sm text-slate-700">
              Your skill levels are determined by analyzing the modules you've completed and the grades you achieved. 
              Higher grades in relevant courses contribute to higher skill proficiency scores. Each skill is mapped from 
              the courses you've taken, and your performance in those courses determines your skill level.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}

