#include <iostream>
#include <string>
#include <locale>
#include <codecvt>
#include <fstream>
#include <vector>
#include <algorithm>
#include <map>
#include <cmath>
#include <tuple>
#include <set>
#include <cassert>
#include <queue>


using namespace std;

typedef pair<int, int> ii;
typedef long long ll;

vector<u16string> VOCAB;
vector<vector<float>> SCORES;
const float NONE = 123456789.0;
const float INF = 999999999.0;

struct Question {
  int length;
  u16string answer;
  Question(u16string answer_) : length(answer_.size()), answer(answer_) {}; 
};
vector<Question> QUESTIONS;

struct ActiveVocab {
  int question_id;
  vector<int> word_ids;
  map<pair<int, char16_t>, float> cache;

  void Restrict(int pos, char16_t c, ActiveVocab* res) const {
    res->question_id = this->question_id;
    for (int word_id : this->word_ids) {
      if (VOCAB[word_id][pos] == c) {
        res->word_ids.push_back(word_id);
      }
    }
  }

  void Create(int question_id) {
    this->question_id = question_id;
    vector<pair<float, int>> scored_word_ids;
    for (int word_id = 0; word_id < VOCAB.size(); ++word_id) {
      if (VOCAB[word_id].size() == QUESTIONS[question_id].length) {
        scored_word_ids.push_back(pair<float, int>(SCORES[question_id][word_id], word_id));
      }
    }
    sort(scored_word_ids.rbegin(), scored_word_ids.rend());
    for (const auto& scored_word_id : scored_word_ids) {
      this->word_ids.push_back(scored_word_id.second);
    }
  }

  float MaxScore() const {
    return SCORES[this->question_id][this->word_ids[0]];
  }

  float MaxRestrictedScore(int pos, char16_t c) {
    const auto& key = pair<int, char16_t>(pos, c);
    auto iter = cache.find(key);
    if (iter == cache.end()) {
      float res = NONE;
      for (int word_id : this->word_ids) {
        if (VOCAB[word_id][pos] == c) {
          res = SCORES[this->question_id][word_id];
          break;
        }
      }
      cache[key] = res;
      return res;
    } else {
      return iter->second;
    }
  }
};
vector<ActiveVocab> ACTIVE_VOCABS;

struct Edge {
  int dst;
  int src_pos;
  int dst_pos;

  Edge() {};
  Edge(int dst_, int src_pos_, int dst_pos_) : dst(dst_), src_pos(src_pos_), dst_pos(dst_pos_) {}
};
vector<vector<Edge>> GRAPH;


void ReadVocab(const string& filename) {
  std::wstring_convert<std::codecvt_utf8_utf16<char16_t>, char16_t> utf16conv;
  ifstream in(filename.c_str());
  string utf8;
  while (in >> utf8) {
    VOCAB.push_back(utf16conv.from_bytes(utf8));
  }
  cout << "Vocab size: " << VOCAB.size() << endl;
}

void ReadQuestions(const string& filename) {
  std::wstring_convert<std::codecvt_utf8_utf16<char16_t>, char16_t> utf16conv;
  ifstream in(filename.c_str());
  // string utf8;
  int length;
  while (in >> length) {
    // u16string answer = utf16conv.from_bytes(utf8);
    u16string answer(length, '*');
    QUESTIONS.push_back(Question(answer));
  }
  cout << "Num questions: " << QUESTIONS.size() << endl;
}

void ReadScores(const string& filename) {
  ifstream in(filename.c_str());
  SCORES.resize(QUESTIONS.size());
  for (int i = 0; i < SCORES.size(); ++i) {
    SCORES[i].resize(VOCAB.size());
    for (int j = 0; j < VOCAB.size(); ++j) {
      SCORES[i][j] = 0;
    }
  }
  int question_id, word_id;
  float score;
  while (in >> question_id >> word_id >> score) {
    SCORES[question_id][word_id] = score;
  }
}

void CreateActiveVocabs() {
  for (int i = 0; i < QUESTIONS.size(); ++i) {
    ACTIVE_VOCABS.push_back(ActiveVocab());
    ACTIVE_VOCABS.back().Create(i);
  }
}

void ReadGraph(const string& filename) {
  ifstream in(filename.c_str());
  int src;
  Edge edge;
  GRAPH.resize(QUESTIONS.size());
  int num_edges = 0;
  while (in >> src >> edge.dst >> edge.src_pos >> edge.dst_pos) {
    GRAPH[src].push_back(edge);
    ++num_edges;
  }
  cout << "Num edges: " << num_edges / 2 << endl;
}


struct Fill {
  int q_index;
  int word_id;

  Fill(int q_index_, int word_id_) : q_index(q_index_), word_id(word_id_) {};
};

bool operator<(const Fill& a, const Fill& b) {
  if (a.q_index == b.q_index) {
    return a.word_id < b.word_id;
  } else {
    return a.q_index < b.q_index;
  }
}


ll GetSolutionHash(vector<Fill> solution) {
  sort(solution.begin(), solution.end());
  
  ll res = 0;
  for (const auto& f : solution) {
    res = res * 11LL + 61LL * (f.q_index + 1LL) + 17LL * (f.word_id + 3LL);
  }
  return res;
}


struct State {
  vector<int> vocab_ids;
  int depth;
  float cost;
  int extra_trials;
  vector<Fill> solution;

  State() {};

  State(const vector<int>& vocab_ids_, int depth_, float cost_, int extra_trials_, const vector<Fill>& solution_) :
    vocab_ids(vocab_ids_), depth(depth_), cost(cost_), extra_trials(extra_trials_), solution(solution_) {};
};


struct PrioritizedState {
  ii priority;
  State state;

  PrioritizedState(const ii& priority_, const State& state_) : priority(priority_), state(state_) {};
};


bool operator<(const PrioritizedState& a, const PrioritizedState& b) {
  return a.priority < b.priority;
}


struct Intersection {
  int c_index;
  int src_pos;
  int dst_pos;

  Intersection(int c_index_, int src_pos_, int dst_pos_) : c_index(c_index_), src_pos(src_pos_), dst_pos(dst_pos_) {};
};


struct VariableCandidate {
  float h_after;
  int word_id;
  float word_score;

  VariableCandidate(float h_after_, int word_id_, float word_score_) : h_after(h_after_), word_id(word_id_), word_score(word_score_) {};
};


bool operator<(const VariableCandidate& a, const VariableCandidate& b) {
  return a.h_after < b.h_after;
}


struct Candidate {
  int c_index;
  int variable_rank;
  float h_after_diff;
  float h_after;
  int word_id;
  float word_score;

  Candidate(int c_index_, int variable_rank_, float h_after_diff_, float h_after_, int word_id_, float word_score_) :
    c_index(c_index_), variable_rank(variable_rank_), h_after_diff(h_after_diff_), h_after(h_after_), word_id(word_id_), word_score(word_score_) {};

  friend ostream& operator<<(ostream& os, const Candidate& c);
};

ostream& operator<<(ostream& os, const Candidate& c) {
  os << c.c_index << ' ' << c.variable_rank << ' ' << c.h_after_diff << ' ' << c.h_after << ' ' << c.word_id << ' ' << c.word_score; 
  return os;
}


bool operator<(const Candidate& a, const Candidate& b) {
  if (a.c_index == b.c_index) {
    return a.variable_rank < b.variable_rank;
  } else {
    return a.h_after_diff < b.h_after_diff;
  }
}


float GetCostUpdate(float word_score) {
  return -log(word_score);
}

auto Q = priority_queue<PrioritizedState>();
float BEST_COST = INF;
vector<int> BEST_COST_ANSWERS;
set<ll> SOLUTION_CACHE;
int MAX_DEPTH;

void Solve(const State& state) {
  const auto& vocab_ids = state.vocab_ids;
  int depth = state.depth;
  float cost = state.cost;
  int extra_trials = state.extra_trials;
  const auto& solution = state.solution;
  if (depth > MAX_DEPTH) {
    // cout << depth << ' ' << solution.size() << endl;
    MAX_DEPTH = depth;
  }

  if (cost > BEST_COST) {
    return;
  }

  if (vocab_ids.empty()) {
    if (cost < BEST_COST) {
      cout << cost << endl;
      BEST_COST = cost;
      for (const auto& f : solution) {
        BEST_COST_ANSWERS[f.q_index] = f.word_id;
      }
    }
    return;
  }


  vector<int> qid2active_id(QUESTIONS.size(), -1);
  vector<float> h_before_scores(vocab_ids.size());
  float h_before = 0;
  float h_before_cost_update = 0;
  for (int c_index = 0; c_index < vocab_ids.size(); ++c_index) {
    const ActiveVocab& c = ACTIVE_VOCABS[vocab_ids[c_index]];
    qid2active_id[c.question_id] = c_index; 
    float max_score = c.MaxScore();
    h_before_scores[c_index] = max_score;
    h_before += max_score;
    h_before_cost_update += GetCostUpdate(max_score);
  }

  if (cost + h_before_cost_update >= BEST_COST) {
    return;
  }

  
  vector<vector<Intersection>> intersections;
  vector<Candidate> candidates;
  for (int c1_index = 0; c1_index < vocab_ids.size(); ++c1_index) {
    const ActiveVocab& c1 = ACTIVE_VOCABS[vocab_ids[c1_index]];

    // Prepare intersections  
    vector<Intersection> variable_intersections;
    for (const auto& e : GRAPH[c1.question_id]) {
      int c2_index = qid2active_id[e.dst];
      if (c2_index != -1) {
        variable_intersections.push_back(
          Intersection(c2_index, e.src_pos, e.dst_pos));
      }
    }
    intersections.push_back(variable_intersections);
    
    // Generate variable candidates  
    vector<VariableCandidate> variable_candidates;
    for (int word_id : c1.word_ids) {
      bool is_valid_word = true;
      float score = SCORES[c1.question_id][word_id];
      float c1_h_after = h_before - h_before_scores[c1_index] + score;
      for (const auto& vi : variable_intersections) {
        int c2_index = vi.c_index;
        ActiveVocab& c2 = ACTIVE_VOCABS[vocab_ids[c2_index]];
        float c2_restricted_score = c2.MaxRestrictedScore(vi.dst_pos, VOCAB[word_id][vi.src_pos]);
        if (c2_restricted_score == NONE) {
          is_valid_word = false;
          break;
        }
        c1_h_after += (c2_restricted_score - h_before_scores[c2_index]);
      }
      if (is_valid_word) {
        variable_candidates.push_back(VariableCandidate(c1_h_after, word_id, score));
      }
    }
    sort(variable_candidates.rbegin(), variable_candidates.rend());
    
    int num_candidates = variable_candidates.size();
    if (num_candidates > 0) {
      for (int i = 0; i + 1 < num_candidates; ++i) {
        float h_after_diff = variable_candidates[i + 1].h_after - variable_candidates[i].h_after;
        candidates.push_back(
          Candidate(c1_index, i, h_after_diff, h_before - variable_candidates[i].h_after,
                    variable_candidates[i].word_id, variable_candidates[i].word_score));
      }
      candidates.push_back(
        Candidate(c1_index, num_candidates - 1, INF,
                  h_before - variable_candidates[num_candidates - 1].h_after,
                  variable_candidates[num_candidates - 1].word_id,
                  variable_candidates[num_candidates - 1].word_score));
    }
  }

  if (candidates.empty()) {
    return;
  }

  stable_sort(candidates.begin(), candidates.end());

  for (int num_trial = 0; num_trial < min(extra_trials, (int)candidates.size()); ++num_trial) {
    const auto& candidate = candidates[num_trial];
    const auto& variable_intersections = intersections[candidate.c_index];

    State new_state;
    new_state.depth = depth + 1;
    new_state.cost = cost + GetCostUpdate(candidate.word_score);
    new_state.extra_trials = extra_trials - num_trial;

    vector<bool> used_indices(vocab_ids.size(), false);
    used_indices[candidate.c_index] = true;
    for (const auto& vi : variable_intersections) {
      int new_vocab_id = ACTIVE_VOCABS.size();
      new_state.vocab_ids.push_back(new_vocab_id);
      ACTIVE_VOCABS.push_back(ActiveVocab());

      int c2_index = vi.c_index;
      ActiveVocab& c2 = ACTIVE_VOCABS[vocab_ids[c2_index]];
      c2.Restrict(vi.dst_pos, VOCAB[candidate.word_id][vi.src_pos], &(ACTIVE_VOCABS[new_vocab_id]));

      used_indices[c2_index] = true;
    }

    for (int c_index = 0; c_index < vocab_ids.size(); ++c_index) {
      if (!used_indices[c_index]) {
        new_state.vocab_ids.push_back(vocab_ids[c_index]);
      }
    }

    const ActiveVocab& c1 = ACTIVE_VOCABS[vocab_ids[candidate.c_index]];
    new_state.solution = state.solution;
    new_state.solution.push_back(Fill(c1.question_id, candidate.word_id));

    ll solution_hash = GetSolutionHash(new_state.solution);
    auto solution_cache_it = SOLUTION_CACHE.find(solution_hash);
    if (solution_cache_it != SOLUTION_CACHE.end()) {
      continue;
    }
    SOLUTION_CACHE.insert(solution_hash);
    Q.push(PrioritizedState(ii(-num_trial, -depth), new_state));
  }
}

int main(int argc, char** argv) {
  ReadVocab(argv[1]);
  ReadQuestions(argv[2]);
  ReadScores(argv[3]);
  ReadGraph(argv[4]);

  BEST_COST_ANSWERS.resize(QUESTIONS.size());
  State s;

  CreateActiveVocabs();
  for (int i = 0; i < ACTIVE_VOCABS.size(); ++i) {
    s.vocab_ids.push_back(i);
  }
  s.depth = 0;
  s.cost = 0;
  s.extra_trials = 5;

  Q.push(PrioritizedState(ii(0, 0), s));

  while (!Q.empty()) {
    const auto s = Q.top();
    Q.pop();
    Solve(s.state);
  }

  if (BEST_COST != INF) {
    std::wstring_convert<std::codecvt_utf8_utf16<char16_t>, char16_t> utf16conv;
    int correct = 0;
    for (int i = 0; i < QUESTIONS.size(); ++i) {
      auto answer = VOCAB[BEST_COST_ANSWERS[i]];
      if (answer == QUESTIONS[i].answer) {
        ++correct;
      } else {
        cout << utf16conv.to_bytes(QUESTIONS[i].answer) << ' ' << utf16conv.to_bytes(answer) << endl;    
      }
    }
    cout << "Acc: " << 1. * correct / QUESTIONS.size() << ". " << correct << " out of " << QUESTIONS.size() << "." << endl;

    ofstream out(argv[5]);
    for (int i = 0; i < QUESTIONS.size(); ++i) {
      auto answer = VOCAB[BEST_COST_ANSWERS[i]];
      out << utf16conv.to_bytes(answer) << endl;
    }
  }

  return 0;
}
