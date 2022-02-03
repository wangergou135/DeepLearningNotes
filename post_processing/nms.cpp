#include <iostream>
#include <vector>
#include <chrono>
#include <algorithm>

using namespace std;

#define AREA(box) ((box.tlbr[3] - box.tlbr[1]) * (box.tlbr[2] - box.tlbr[0]))

struct box {
    float tlbr[4];
    float score;
};

void print_boxes(vector<box>& boxes)
{
    cout << "size:" << boxes.size() << endl;
    for (auto& box : boxes) {
        cout << box.tlbr[0] << ", " << box.tlbr[1] << ", " << box.tlbr[2] 
            << ", "<< box.tlbr[3] << ", "<< box.score << endl;
    }
    cout << endl;
}

float iou(box& a, box& b)
{
    static float tlbr[4];
    static float intersection;
    tlbr[0] = a.tlbr[0] > b.tlbr[0] ? a.tlbr[0] : b.tlbr[0];
    tlbr[1] = a.tlbr[1] > b.tlbr[1] ? a.tlbr[1] : b.tlbr[1];
    tlbr[2] = a.tlbr[2] < b.tlbr[2] ? a.tlbr[2] : b.tlbr[2];
    tlbr[3] = a.tlbr[3] < b.tlbr[3] ? a.tlbr[3] : b.tlbr[3];

    if (tlbr[0] >= tlbr[2] || tlbr[1] >= tlbr[3]) {
        return 0;
    }

    intersection = (tlbr[2] - tlbr[0]) * (tlbr[3] - tlbr[1]);

    return intersection / (AREA(a) + AREA(b) - intersection); 
}

int nms(vector<box>& boxes, float threshold)
{
    int cur_index = 0;
    int invalid_index = boxes.size() - 1;
    int i = 0;
    
    for (i = 1; i < boxes.size(); i++) {
        if (boxes[i].score > boxes[0].score) {
            swap(boxes[0], boxes[i]);
        }
    }

    i = 1;
    while (i < boxes.size()) {
        cout << i << ", " << cur_index << "," << invalid_index << endl;
        if (iou(boxes[cur_index], boxes[i]) >= threshold) {
            swap(boxes[i], boxes[invalid_index]);
            invalid_index--;
        } else {
            i++;
        }
        if (invalid_index == i) {
            i = ++cur_index + 1;
        }
    }

    return invalid_index;
}

int main()
{
    vector<box> boxes= {    
        {100,100,210,210,0.72},
        {250,250,420,420,0.8 },
        {220,220,320,330,0.92},
        {100,100,210,210,0.73},
        {230,240,325,330,0.81},
        {220,230,315,340,0.9 } 
                        };
    
    cout << "result boxes:" << nms(boxes, 0.7) << endl;

    print_boxes(boxes);
    
    return 0;
}