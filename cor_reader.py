import argparse
import collections

from utils import cornell, data


if __name__  == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--genre", default='', help="Genre to show dialogs from")
    parser.add_argument("--show-genres", action='store_true', default=False, help="Display genres stats")
    parser.add_argument("--show-dials", action='store_true', default=False, help="Display dialogs")
    parser.add_argument("--show-train", action='store_true', default=False, help="Display training pairs")
    parser.add_argument("--show-dict-freq", action='store_true', default=False, help="Display dictionary frequency")
    args = parser.parse_args()

    if args.show_genres:
        # Khởi tạo collection
        genre_counts = collections.Counter()
        genres = cornell.read_genres(cornell.DATA_DIR)
        for movie, g_list in genres.items():
            # print("Movies is", g_list)
            for g in g_list:
                genre_counts[g] += 1
        print("Genres:")
        for g, count in genre_counts.most_common():
            print("%s: %d" % (g, count))

