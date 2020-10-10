import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run experiments for the JJS229 project.')
    parser.add_argument('--all', action='store_true', help='Run all experiments.')
    args = parser.parse_args()

    print(f'Arguments: {args}')

