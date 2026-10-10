"""R1 non-oracle localization; independent artifacts, no R0 modifications."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.cirlid.config import Config

METHODS = ("relation", "local_difference", "position_template", "broken_context")
SEMANTIC_CONTROL = "local_difference_semantic"
SEMANTIC_CONTROL_METHODS = (*METHODS, SEMANTIC_CONTROL)


def methods_for(config: 'Config') -> tuple[str, ...]:
    """Keep the original four-method protocol and the supplement distinct."""
    from src.cirlid.config import ConfigError

    profiles = {
        'r1_non_oracle_semantic_relation_v1': ('R1-non-oracle-v1.0', METHODS),
        'r1_non_oracle_semantic_relation_semantic_control_v1': (
            'R1-non-oracle-semantic-control-v1.0', SEMANTIC_CONTROL_METHODS),
    }
    profile = profiles.get(config.get('protocol.namespace'))
    if profile is None:
        raise ConfigError('unknown R1 protocol namespace')
    version, methods = profile
    if config.get('protocol.version') != version:
        raise ConfigError('R1 namespace/version mismatch')
    selected = config.get('scoring.methods')
    if selected is None and methods == METHODS:
        return methods  # Original v1 configuration has no scoring section.
    if selected != list(methods):
        raise ConfigError('scoring.methods must match the registered R1 profile exactly')
    if methods == SEMANTIC_CONTROL_METHODS:
        expected = 'copy_aggregated_local_then_zero_outside_query_relation_token_support'
        if config.get('scoring.semantic_control_definition') != expected:
            raise ConfigError('semantic local control must use the relation token support')
    return methods
