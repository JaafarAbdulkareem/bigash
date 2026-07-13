import 'package:bigash/domain/currency/reposition_guidance.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  test('three hints are followed by one failure statement then silence', () {
    final RepositionGuidancePolicy policy = RepositionGuidancePolicy();

    expect(policy.next(), RepositionGuidanceDecision.repeatHint);
    expect(policy.next(), RepositionGuidanceDecision.repeatHint);
    expect(policy.next(), RepositionGuidanceDecision.repeatHint);
    expect(policy.spokenHints, 3);
    expect(policy.next(), RepositionGuidanceDecision.announceFailure);
    expect(policy.next(), RepositionGuidanceDecision.silent);
  });

  test('reset starts a fresh scan guidance sequence', () {
    final RepositionGuidancePolicy policy = RepositionGuidancePolicy();
    for (int i = 0; i < 4; i++) {
      policy.next();
    }
    policy.reset();

    expect(policy.spokenHints, 0);
    expect(policy.next(), RepositionGuidanceDecision.repeatHint);
  });
}
