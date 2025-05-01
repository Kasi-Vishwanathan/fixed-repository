from tree_sitter import Language, Parser
import sys

# Define platform-specific shared library extension
output_file = 'my-languages' + ('.dll' if sys.platform == 'win32' else '.so')

# Update these paths to point to your tree-sitter language repositories
language_repositories = [
    # Replace with actual paths to cloned repositories
    'vendor/tree-sitter-go',
    'vendor/tree-sitter-javascript',
    'vendor/tree-sitter-python',
    'vendor/tree-sitter-php',
    'vendor/tree-sitter-java',
    'vendor/tree-sitter-ruby',
    'vendor/tree-sitter-c-sharp',
]

Language.build_library(
    output_file,
    language_repositories
)