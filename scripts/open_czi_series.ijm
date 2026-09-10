// open_czi_series.ijm
// Opens series 1 from each .czi file in a user-selected directory and saves as TIFF in the same directory.

input_dir = getDirectory("Choose input directory containing .czi files");
output_dir = input_dir;

list = getFileList(input_dir);

for (i = 0; i < list.length; i++) {
    if (endsWith(list[i], ".czi")) {
        path = input_dir + list[i];

        run("Bio-Formats Importer", "open=[" + path + "] autoscale color_mode=Default view=Hyperstack stack_order=XYCZT series_1");

        stem = replace(list[i], ".czi", "");
        saveAs("Tiff", output_dir + stem + "_s1.tif");
        close();
    }
}
