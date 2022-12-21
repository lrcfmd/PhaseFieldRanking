from ..ranking_phase_fields.__main__ import main
import sys
if __name__=="__main__":
    try:
        input_file = sys.argv[1]
    except Exception:
        print("Usage: python ranking_phase_fields.py <input_file>")
        print("Reading default settings from rpp.input")
        input_file = 'rpp.input'
    main(input_file)
