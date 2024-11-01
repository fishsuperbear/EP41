/*
   Copyright (C) 2006-2016 Con Kolivas
   Copyright (C) 2011 Peter Hyman
   Copyright (C) 1998-2003 Andrew Tridgell

   This program is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation; either version 2 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program. If not, see <http://www.gnu.org/licenses/>.
*/
/* lrzip compression - main program */

#ifdef HAVE_CONFIG_H
# include "config.h"
#endif

#include <signal.h>
#ifdef HAVE_UNISTD_H
# include <unistd.h>
#endif
#ifdef HAVE_SYS_TIME_H
# include <sys/time.h>
#endif
#ifdef HAVE_SYS_RESOURCE_H
# include <sys/resource.h>
#endif
#ifdef HAVE_SYS_TYPES_H
# include <sys/types.h>
#endif
#ifdef HAVE_SYS_STAT_H
# include <sys/stat.h>
#endif

#include <termios.h>
#ifdef HAVE_ENDIAN_H
# include <endian.h>
#elif HAVE_SYS_ENDIAN_H
# include <sys/endian.h>
#endif
#ifdef HAVE_ARPA_INET_H
# include <arpa/inet.h>
#endif

#include <dirent.h>
#include <getopt.h>
#include <libgen.h>

#include "rzip.h"
#include "lrzip_core.h"
#include "util.h"
#include "stream.h"

/* needed for CRC routines */
#include "lzma/C/7zCrc.h"

#define MAX_PATH_LEN 4096

static rzip_control base_control, local_control, *control;

static void usage(bool compat)
{
	print_output("lrz%s version %s\n", compat ? "" : "ip", PACKAGE_VERSION);
	print_output("Copyright (C) Con Kolivas 2006-2016\n");
	print_output("Based on rzip ");
	print_output("Copyright (C) Andrew Tridgell 1998-2003\n\n");
	print_output("Usage: lrz%s [options] <file...>\n", compat ? "" : "ip");
	print_output("General options:\n");
	if (compat) {
		print_output("	-c, --stdout		output to STDOUT\n");
		print_output("	-C, --check		check integrity of file written on decompression\n");
	} else
		print_output("	-c, -C, --check		check integrity of file written on decompression\n");
	print_output("	-d, --decompress	decompress\n");
	print_output("	-e, --encrypt		password protected sha512/aes128 encryption on compression\n");
	print_output("	-h, -?, --help		show help\n");
	print_output("	-H, --hash		display md5 hash integrity information\n");
	print_output("	-i, --info		show compressed file information\n");
	if (compat) {
		print_output("	-L, --license		display software version and license\n");
		print_output("	-P, --progress		show compression progress\n");
	} else
		print_output("	-q, --quiet		don't show compression progress\n");
	print_output("	-r, --recursive		operate recursively on directories\n");
	print_output("	-t, --test		test compressed file integrity\n");
	print_output("	-v[v%s], --verbose	Increase verbosity\n", compat ? "v" : "");
	print_output("	-V, --version		show version\n");
	print_output("Options affecting output:\n");
	if (!compat)
		print_output("	-D, --delete		delete existing files\n");
	print_output("	-f, --force		force overwrite of any existing files\n");
	if (compat)
		print_output("	-k, --keep		don't delete source files on de/compression\n");
	print_output("	-K, --keep-broken	keep broken or damaged output files\n");
	print_output("	-o, --outfile filename	specify the output file name and/or path\n");
	print_output("	-O, --outdir directory	specify the output directory when -o is not used\n");
	print_output("	-S, --suffix suffix	specify compressed suffix (default '.lrz')\n");
	print_output("Options affecting compression:\n");
	print_output("	--lzma			lzma compression (default)\n");
	print_output("	-b, --bzip2		bzip2 compression\n");
	print_output("	-g, --gzip		gzip compression using zlib\n");
	print_output("	-l, --lzo		lzo compression (ultra fast)\n");
	print_output("	-n, --no-compress	no backend compression - prepare for other compressor\n");
	print_output("	-z, --zpaq		zpaq compression (best, extreme compression, extremely slow)\n");
	print_output("Low level options:\n");
	if (compat) {
		print_output("	-1 .. -9		set lzma/bzip2/gzip compression level (1-9, default 7)\n");
		print_output("	--fast			alias for -1\n");
		print_output("	--best			alias for -9\n");
	}
	if (!compat)
		print_output("	-L, --level level	set lzma/bzip2/gzip compression level (1-9, default 7)\n");
	print_output("	-N, --nice-level value	Set nice value to value (default %d)\n", compat ? 0 : 19);
	print_output("	-p, --threads value	Set processor count to override number of threads\n");
	print_output("	-m, --maxram size	Set maximim available ram in hundreds of MB\n");
	print_output("				overrides detected ammount of available ram\n");
	print_output("	-T, --threshold		Disable LZO compressibility testing\n");
	print_output("	-U, --unlimited		Use unlimited window size beyond ramsize (potentially much slower)\n");
	print_output("	-w, --window size	maximum compression window in hundreds of MB\n");
	print_output("				default chosen by heuristic dependent on ram and chosen compression\n");
	print_output("\nLRZIP=NOCONFIG environment variable setting can be used to bypass lrzip.conf.\n");
	print_output("TMP environment variable will be used for storage of temporary files when needed.\n");
	print_output("TMPDIR may also be stored in lrzip.conf file.\n");
	print_output("\nIf no filenames or \"-\" is specified, stdin/out will be used.\n");

}

static void license(void)
{
	print_output("lrz version %s\n", PACKAGE_VERSION);
	print_output("Copyright (C) Con Kolivas 2006-2016\n");
	print_output("Based on rzip ");
	print_output("Copyright (C) Andrew Tridgell 1998-2003\n\n");
	print_output("This is free software.  You may redistribute copies of it under the terms of\n");
	print_output("the GNU General Public License <http://www.gnu.org/licenses/gpl.html>.\n");
	print_output("There is NO WARRANTY, to the extent permitted by law.\n");
}

static void sighandler(int sig __UNUSED__)
{
	struct termios termios_p;

	/* Make sure we haven't died after disabling stdin echo */
	tcgetattr(fileno(stdin), &termios_p);
	termios_p.c_lflag |= ECHO;
	tcsetattr(fileno(stdin), 0, &termios_p);

	unlink_files(control);
	//exit(0);
}

static void show_summary(void)
{
	/* OK, if verbosity set, print summary of options selected */
	if (!INFO) {
		if (!TEST_ONLY)
			print_verbose("The following options are in effect for this %s.\n",
				      DECOMPRESS ? "DECOMPRESSION" : "COMPRESSION");
		print_verbose("Threading is %s. Number of CPUs detected: %d\n", control->threads > 1? "ENABLED" : "DISABLED",
			      control->threads);
		print_verbose("Detected %lld bytes ram\n", control->ramsize);
		print_verbose("Compression level %d\n", control->compression_level);
		print_verbose("Nice Value: %d\n", control->nice_val);
		print_verbose("Show Progress\n");
		print_maxverbose("Max ");
		print_verbose("Verbose\n");
		if (FORCE_REPLACE)
			print_verbose("Overwrite Files\n");
		if (!KEEP_FILES)
			print_verbose("Remove input files on completion\n");
		if (control->outdir)
			print_verbose("Output Directory Specified: %s\n", control->outdir);
		else if (control->outname)
			print_verbose("Output Filename Specified: %s\n", control->outname);
		if (TEST_ONLY)
			print_verbose("Test file integrity\n");
		if (control->tmpdir)
			print_verbose("Temporary Directory set as: %s\n", control->tmpdir);

		/* show compression options */
		if (!DECOMPRESS && !TEST_ONLY) {
			print_verbose("Compression mode is: ");
			if (LZMA_COMPRESS)
				print_verbose("LZMA. LZO Compressibility testing %s\n", (LZO_TEST? "enabled" : "disabled"));
			else if (LZO_COMPRESS)
				print_verbose("LZO\n");
			else if (BZIP2_COMPRESS)
				print_verbose("BZIP2. LZO Compressibility testing %s\n", (LZO_TEST? "enabled" : "disabled"));
			else if (ZLIB_COMPRESS)
				print_verbose("GZIP\n");
			else if (ZPAQ_COMPRESS)
				print_verbose("ZPAQ. LZO Compressibility testing %s\n", (LZO_TEST? "enabled" : "disabled"));
			else if (NO_COMPRESS)
				print_verbose("RZIP pre-processing only\n");
			if (control->window)
				print_verbose("Compression Window: %lld = %lldMB\n", control->window, control->window * 100ull);
			/* show heuristically computed window size */
			if (!control->window && !UNLIMITED) {
				i64 temp_chunk, temp_window;

				if (STDOUT || STDIN)
					temp_chunk = control->maxram;
				else
					temp_chunk = control->ramsize * 2 / 3;
				temp_window = temp_chunk / (100 * 1024 * 1024);
				print_verbose("Heuristically Computed Compression Window: %lld = %lldMB\n", temp_window, temp_window * 100ull);
			}
			if (UNLIMITED)
				print_verbose("Using Unlimited Window size\n");
		}
		if (!DECOMPRESS && !TEST_ONLY)
			print_maxverbose("Storage time in seconds %lld\n", control->secs);
		if (ENCRYPT)
			print_maxverbose("Encryption hash loops %lld\n", control->encloops);
	}
}

static struct option long_options[] = {
	{"bzip2",	no_argument,	0,	'b'}, /* 0 */
	{"check",	no_argument,	0,	'c'},
	{"check",	no_argument,	0,	'C'},
	{"decompress",	no_argument,	0,	'd'},
	{"delete",	no_argument,	0,	'D'},
	{"encrypt",	no_argument,	0,	'e'}, /* 5 */
	{"force",	no_argument,	0,	'f'},
	{"gzip",	no_argument,	0,	'g'},
	{"help",	no_argument,	0,	'h'},
	{"hash",	no_argument,	0,	'H'},
	{"info",	no_argument,	0,	'i'}, /* 10 */
	{"keep-broken",	no_argument,	0,	'k'},
	{"keep-broken",	no_argument,	0,	'K'},
	{"lzo",		no_argument,	0,	'l'},
	{"lzma",       	no_argument,	0,	'/'},
	{"level",	optional_argument,	0,	'L'}, /* 15 */
	{"license",	no_argument,	0,	'L'},
	{"maxram",	required_argument,	0,	'm'},
	{"no-compress",	no_argument,	0,	'n'},
	{"nice-level",	required_argument,	0,	'N'},
	{"outfile",	required_argument,	0,	'o'},
	{"outdir",	required_argument,	0,	'O'}, /* 20 */
	{"threads",	required_argument,	0,	'p'},
	{"progress",	no_argument,	0,	'P'},
	{"quiet",	no_argument,	0,	'q'},
	{"recursive",	no_argument,	0,	'r'},
	{"suffix",	required_argument,	0,	'S'},
	{"test",	no_argument,	0,	't'},  /* 25 */
	{"threshold",	required_argument,	0,	'T'},
	{"unlimited",	no_argument,	0,	'U'},
	{"verbose",	no_argument,	0,	'v'},
	{"version",	no_argument,	0,	'V'},
	{"window",	required_argument,	0,	'w'},  /* 30 */
	{"zpaq",	no_argument,	0,	'z'},
	{"fast",	no_argument,	0,	'1'},
	{"best",	no_argument,	0,	'9'},
	{0,	0,	0,	0},
};

static void set_stdout(struct rzip_control *control)
{
	control->flags |= FLAG_STDOUT;
	control->outFILE = stdout;
	control->msgout = stderr;
	register_outputfile(control, control->msgout);
}

/* Recursively enter all directories, adding all regular files to the dirlist array */
static void recurse_dirlist(char *indir, char **dirlist, int *entries)
{
	char fname[MAX_PATH_LEN];
	struct stat istat;
	struct dirent *dp;
	DIR *dirp;

	dirp = opendir(indir);
	if (unlikely(!dirp))
		failure("Unable to open directory %s\n", indir);
	while ((dp = readdir(dirp)) != NULL) {
		if (!strcmp(dp->d_name, ".") || !strcmp(dp->d_name, ".."))
			continue;
		sprintf(fname, "%s/%s", indir, dp->d_name);
		if (unlikely(stat(fname, &istat)))
			failure("Unable to stat file %s\n", fname);
		if (S_ISDIR(istat.st_mode)) {
			recurse_dirlist(fname, dirlist, entries);
			continue;
		}
		if (!S_ISREG(istat.st_mode)) {
			print_err("Not regular file %s\n", fname);
			continue;
		}
		print_maxverbose("Added file %s\n", fname);
		*dirlist = realloc(*dirlist, MAX_PATH_LEN * (*entries + 1));
		strcpy(*dirlist + MAX_PATH_LEN * (*entries)++, fname);
	}
	closedir(dirp);
}

static const char *loptions = "bcCdDefghHiKlL:nN:o:O:p:PqrS:tTUm:vVw:z?";
static const char *coptions = "bcCdefghHikKlLnN:o:O:p:PrS:tTUm:vVw:z?123456789";

static int LrzipMain(int argc, char *argv[])
{
	optind = 1;

	bool lrzcat = false, compat = false, recurse = false;
	struct timeval start_time, end_time;
	struct sigaction handler;
	double seconds,total_time; // for timers
	int c, i;
	int hours,minutes;
	extern int optind;
	char *eptr, *av; /* for environment */

        control = &base_control;

	initialise_control(control);

	av = basename(argv[0]);
	if (!strcmp(av, "lrunzip"))
		control->flags |= FLAG_DECOMPRESS;
	else if (!strcmp(av, "lrzcat")) {
		control->flags |= FLAG_DECOMPRESS | FLAG_STDOUT;
		lrzcat = true;
	} else if (!strcmp(av, "lrz")) {
		/* Called in gzip compatible command line mode */
		control->flags &= ~FLAG_SHOW_PROGRESS;
		control->nice_val = 0;
		control->flags &= ~FLAG_KEEP_FILES;
		compat = true;
		long_options[1].name = "stdout";
		long_options[11].name = "keep";
	}

	/* generate crc table */
	CrcGenerateTable();

	/* Get Preloaded Defaults from lrzip.conf
	 * Look in ., $HOME/.lrzip/, /etc/lrzip.
	 * If LRZIP=NOCONFIG is set, then ignore config
	 */
	eptr = getenv("LRZIP");
	if (eptr == NULL)
		read_config(control);
	else if (!strstr(eptr,"NOCONFIG"))
		read_config(control);

	while ((c = getopt_long(argc, argv, compat ? coptions : loptions, long_options, &i)) != -1) {
		switch (c) {
		case 'b':
			if (control->flags & FLAG_NOT_LZMA)
				failure("Can only use one of -l, -b, -g, -z or -n\n");
			control->flags |= FLAG_BZIP2_COMPRESS;
			break;
		case 'c':
			if (compat) {
				control->flags |= FLAG_KEEP_FILES;
				set_stdout(control);
				break;
			}
		case 'C':
			control->flags |= FLAG_CHECK;
			control->flags |= FLAG_HASH;
			break;
		case 'd':
			control->flags |= FLAG_DECOMPRESS;
			break;
		case 'D':
			control->flags &= ~FLAG_KEEP_FILES;
			break;
		case 'e':
			control->flags |= FLAG_ENCRYPT;
			break;
		case 'f':
			control->flags |= FLAG_FORCE_REPLACE;
			break;
		case 'g':
			if (control->flags & FLAG_NOT_LZMA)
				failure("Can only use one of -l, -b, -g, -z or -n\n");
			control->flags |= FLAG_ZLIB_COMPRESS;
			break;
		case 'h':
		case '?':
			usage(compat);
			return -1;
		case 'H':
			control->flags |= FLAG_HASH;
			break;
		case 'i':
			control->flags |= FLAG_INFO;
			break;
		case 'k':
			if (compat) {
				control->flags |= FLAG_KEEP_FILES;
				break;
			}
		case 'K':
			control->flags |= FLAG_KEEP_BROKEN;
			break;
		case 'l':
			if (control->flags & FLAG_NOT_LZMA)
				failure("Can only use one of -l, -b, -g, -z or -n\n");
			control->flags |= FLAG_LZO_COMPRESS;
			break;
		case 'L':
			if (compat) {
				license();
				exit(0);
			}
			control->compression_level = atoi(optarg);
			if (control->compression_level < 1 || control->compression_level > 9)
				failure("Invalid compression level (must be 1-9)\n");
			break;
		case 'm':
			control->ramsize = atol(optarg) * 1024 * 1024 * 100;
			break;
		case 'n':
			if (control->flags & FLAG_NOT_LZMA)
				failure("Can only use one of -l, -b, -g, -z or -n\n");
			control->flags |= FLAG_NO_COMPRESS;
			break;
		case 'N':
			control->nice_val = atoi(optarg);
			if (control->nice_val < -20 || control->nice_val > 19)
				failure("Invalid nice value (must be -20..19)\n");
			break;
		case 'o':
			if (control->outdir)
				failure("Cannot have -o and -O together\n");
			if (unlikely(STDOUT))
				failure("Cannot specify an output filename when outputting to stdout\n");
			control->outname = optarg;
			control->suffix = "";
			break;
		case 'O':
			if (control->outname)	/* can't mix -o and -O */
				failure("Cannot have options -o and -O together\n");
			if (unlikely(STDOUT))
				failure("Cannot specify an output directory when outputting to stdout\n");
			control->outdir = malloc(strlen(optarg) + 2);
			if (control->outdir == NULL)
				fatal("Failed to allocate for outdir\n");
			strcpy(control->outdir,optarg);
			if (strcmp(optarg+strlen(optarg) - 1, "/")) 	/* need a trailing slash */
				strcat(control->outdir, "/");
			break;
		case 'p':
			control->threads = atoi(optarg);
			if (control->threads < 1)
				failure("Must have at least one thread\n");
			break;
		case 'P':
			control->flags |= FLAG_SHOW_PROGRESS;
			break;
		case 'q':
			control->flags &= ~FLAG_SHOW_PROGRESS;
			break;
		case 'r':
			recurse = true;
			break;
		case 'S':
			if (control->outname)
				failure("Specified output filename already, can't specify an extension.\n");
			if (unlikely(STDOUT))
				failure("Cannot specify a filename suffix when outputting to stdout\n");
			control->suffix = optarg;
			break;
		case 't':
			if (control->outname)
				failure("Cannot specify an output file name when just testing.\n");
			if (compat)
				control->flags |= FLAG_KEEP_FILES;
			if (!KEEP_FILES)
				failure("Doubt that you want to delete a file when just testing.\n");
			control->flags |= FLAG_TEST_ONLY;
			break;
		case 'T':
			control->flags &= ~FLAG_THRESHOLD;
			break;
		case 'U':
			control->flags |= FLAG_UNLIMITED;
			break;
		case 'v':
			/* set verbosity flag */
			if (!(control->flags & FLAG_SHOW_PROGRESS))
				control->flags |= FLAG_SHOW_PROGRESS;
			else if (!(control->flags & FLAG_VERBOSITY) && !(control->flags & FLAG_VERBOSITY_MAX))
				control->flags |= FLAG_VERBOSITY;
			else if ((control->flags & FLAG_VERBOSITY)) {
				control->flags &= ~FLAG_VERBOSITY;
				control->flags |= FLAG_VERBOSITY_MAX;
			}
			break;
		case 'V':
			print_output("lrzip version %s\n", PACKAGE_VERSION);
			exit(0);
			break;
		case 'w':
			control->window = atol(optarg);
			break;
		case 'z':
			if (control->flags & FLAG_NOT_LZMA)
				failure("Can only use one of -l, -b, -g, -z or -n\n");
			control->flags |= FLAG_ZPAQ_COMPRESS;
			break;
		case '1':
		case '2':
		case '3':
		case '4':
		case '5':
		case '6':
		case '7':
		case '8':
		case '9':
			control->compression_level = c - '0';
			break;
		}
	}

	argc -= optind;
	argv += optind;

	if (control->outname) {
		if (argc > 1)
			failure("Cannot specify output filename with more than 1 file\n");
		if (recurse)
			failure("Cannot specify output filename with recursive\n");
	}

	if (VERBOSE && !SHOW_PROGRESS) {
		print_err("Cannot have -v and -q options. -v wins.\n");
		control->flags |= FLAG_SHOW_PROGRESS;
	}

	if (UNLIMITED && control->window) {
		print_err("If -U used, cannot specify a window size with -w.\n");
		control->window = 0;
	}

	if (argc < 1)
		control->flags |= FLAG_STDIN;

	if (UNLIMITED && STDIN) {
		print_err("Cannot have -U and stdin, unlimited mode disabled.\n");
		control->flags &= ~FLAG_UNLIMITED;
	}

	setup_overhead(control);

	/* Set the main nice value to half that of the backend threads since
	 * the rzip stage is usually the rate limiting step */
	if (control->nice_val > 0 && !NO_COMPRESS) {
		if (unlikely(setpriority(PRIO_PROCESS, 0, control->nice_val / 2) == -1))
			print_err("Warning, unable to set nice value\n");
	} else {
		if (unlikely(setpriority(PRIO_PROCESS, 0, control->nice_val) == -1))
			print_err("Warning, unable to set nice value\n");
	}

	/* One extra iteration for the case of no parameters means we will default to stdin/out */
	for (i = 0; i <= argc; i++) {
		char *dirlist = NULL, *infile = NULL;
		int direntries = 0, curentry = 0;

		if (i < argc)
			infile = argv[i];
		else if (!(i == 0 && STDIN))
			break;
		if (infile) {
			if ((strcmp(infile, "-") == 0))
				control->flags |= FLAG_STDIN;
			else {
				bool isdir = false;
				struct stat istat;

				if (unlikely(stat(infile, &istat)))
					failure("Failed to stat %s\n", infile);
				isdir = S_ISDIR(istat.st_mode);
				if (!recurse && (isdir || !S_ISREG(istat.st_mode))) {
					failure("lrzip only works directly on regular FILES.\n"
					"Use -r recursive, lrztar or pipe through tar for compressing directories.\n");
				}
				if (recurse && !isdir)
					failure("%s not a directory, -r recursive needs a directory\n", infile);
			}
		}

		if (recurse) {
			if (unlikely(STDIN || STDOUT))
				failure("Cannot use -r recursive with STDIO\n");
			recurse_dirlist(infile, &dirlist, &direntries);
		}

		if (INFO && STDIN)
			failure("Will not get file info from STDIN\n");
recursion:
		if (recurse) {
			if (curentry >= direntries) {
				infile = NULL;
				continue;
			}
			infile = dirlist + MAX_PATH_LEN * curentry++;
		}
		control->infile = infile;

		/* If no output filename is specified, and we're using
		 * stdin, use stdout */
		if ((control->outname && (strcmp(control->outname, "-") == 0)) ||
		    (!control->outname && STDIN) || lrzcat)
				set_stdout(control);

		if (lrzcat) {
			control->msgout = stderr;
			control->outFILE = stdout;
			register_outputfile(control, control->msgout);
		}

		if (!STDOUT) {
			control->msgout = stdout;
			register_outputfile(control, control->msgout);
		}

		if (STDIN)
			control->inFILE = stdin;
		/* Implement signal handler only once flags are set */
		// sigemptyset(&handler.sa_mask);
		// handler.sa_flags = 0;
		// handler.sa_handler = &sighandler;
		// sigaction(SIGTERM, &handler, 0);
		// sigaction(SIGINT, &handler, 0);

		if (!FORCE_REPLACE) {
			if (STDIN && isatty(fileno((FILE *)stdin))) {
				print_err("Will not read stdin from a terminal. Use -f to override.\n");
				usage(compat);
				exit (1);
			}
			if (!TEST_ONLY && STDOUT && isatty(fileno((FILE *)stdout)) && !compat) {
				print_err("Will not write stdout to a terminal. Use -f to override.\n");
				usage(compat);
				exit (1);
			}
		}

		if (CHECK_FILE) {
			if (!DECOMPRESS) {
				print_err("Can only check file written on decompression.\n");
				control->flags &= ~FLAG_CHECK;
			} else if (STDOUT) {
				print_err("Can't check file written when writing to stdout. Checking disabled.\n");
				control->flags &= ~FLAG_CHECK;
			}
		}

		setup_ram(control);
		show_summary();

		gettimeofday(&start_time, NULL);

		if (unlikely(STDIN && ENCRYPT))
			failure("Unable to work from STDIN while reading password\n");

		memcpy(&local_control, &base_control, sizeof(rzip_control));
		if (DECOMPRESS || TEST_ONLY)
			decompress_file(&local_control);
		else if (INFO)
			get_fileinfo(&local_control);
		else
			compress_file(&local_control);

		/* compute total time */
		gettimeofday(&end_time, NULL);
		total_time = (end_time.tv_sec + (double)end_time.tv_usec / 1000000) -
			      (start_time.tv_sec + (double)start_time.tv_usec / 1000000);
		hours = (int)total_time / 3600;
		minutes = (int)(total_time / 60) % 60;
		seconds = total_time - hours * 3600 - minutes * 60;
		if (!INFO)
			print_progress("Total time: %02d:%02d:%05.2f\n", hours, minutes, seconds);
		if (recurse)
			goto recursion;
	}

	return 0;
}


int lrunzip(char *source_file, char *dest_dir)
{
	int argc = 0;
	char *argv[20];
	int ret = -1;

	do {
		if (strstr(source_file, ".tar.lrz") == NULL) {
			ret = -1;
			break;
		}
		if (dest_dir == NULL) {
			dest_dir = ".";
		}
		
		argv[argc++] = "lrzip";
		argv[argc++] = "-d"; // 解压
		argv[argc++] = "-f"; // 强制覆盖
		argv[argc++] = "-O"; // 输出到指定目录
		argv[argc++] = dest_dir;
		argv[argc++] = source_file;
		if (LrzipMain(argc, argv) < 0) {
			ret = -1;
			break;
		}

		//进行tar，并删除.tar文件
		char cmd[1000];
		char destfile[1000];
		char *p = NULL;
		memset(cmd, 0, sizeof(cmd));
		memset(destfile, 0, sizeof(destfile));
		
		if (strlen(dest_dir) > sizeof(destfile) - 1) {
			ret = -1;
			break;
		}
		strcpy(destfile, dest_dir);
		if ((p = strrchr(source_file, '/')) == NULL) {
			p = source_file;
			strcat(destfile, "/");
		}
		strcat(destfile, p);
		if ((p = strstr(destfile, ".lrz")) != NULL) {
			p[0] = 0x00;
		}

		strcat(cmd, "tar -xf ");
		strcat(cmd, destfile);
		strcat(cmd, " -C "); // tar到指定目录
		strcat(cmd, dest_dir);
		printf("system: %s\n", cmd);
		if (system(cmd) < 0) {
			printf("error cmd: %s\n", cmd);
			ret = -1;
			break;
		}

		memset(cmd, 0, sizeof(cmd));
		strcat(cmd, "rm ");
		strcat(cmd, destfile);
		printf("system: %s\n", cmd);
		if (system(cmd) < 0) {
			printf("error: %s\n", cmd);
			ret = -1;
			break;
		}
	} while(0);

	//todo: lrzip对内存管理比较混乱，很多地方未做释放，在这边统一释放也不合理，可能出现重复释放的问题
	// if (base_control.tmpdir) {
	// 	free(base_control.tmpdir);
	// 	base_control.tmpdir = NULL;
	// }
	// if (base_control.suffix) {
	// 	free(base_control.suffix);
	// 	base_control.suffix = NULL;
	// }
	// if (base_control.outdir) {
	// 	free(base_control.outdir);
	// 	base_control.outdir = NULL;
	// }
	// if (base_control.tmp_outbuf) {
	// 	free(base_control.tmp_outbuf);
	// 	base_control.tmp_outbuf = NULL;
	// }
	// if (base_control.outfile) {
	// 	free(base_control.outfile);
	// 	base_control.outfile = NULL;
	// }
	// if (local_control.tmp_outbuf) {
	// 	free(local_control.tmp_outbuf);
	// 	local_control.tmp_outbuf = NULL;
	// }
	// if (local_control.outfile) {
	// 	free(local_control.outfile);
	// 	local_control.outfile = NULL;
	// }
	ret = 0;

	return ret;
}
