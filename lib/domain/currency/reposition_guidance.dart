enum RepositionGuidanceDecision { repeatHint, announceFailure, silent }

/// Limits repetitive distance guidance for a blind user.
class RepositionGuidancePolicy {
  RepositionGuidancePolicy({this.maximumRepeatedHints = 3});

  final int maximumRepeatedHints;
  int _spokenHints = 0;
  bool _failureAnnounced = false;

  int get spokenHints => _spokenHints;

  RepositionGuidanceDecision next() {
    if (_failureAnnounced) return RepositionGuidanceDecision.silent;
    if (_spokenHints < maximumRepeatedHints) {
      _spokenHints++;
      return RepositionGuidanceDecision.repeatHint;
    }
    _failureAnnounced = true;
    return RepositionGuidanceDecision.announceFailure;
  }

  void reset() {
    _spokenHints = 0;
    _failureAnnounced = false;
  }
}
