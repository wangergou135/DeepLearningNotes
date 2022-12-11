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
void print_ious(vector<box>& boxes)
{
    for (int i = 0; i < boxes.size(); i++)
    {

        
        for (int j = 0; j < boxes.size(); j++)
        {
            cout << iou(boxes[i], boxes[j]) << ",\t" ;
        }
        cout << endl;
    }
}
// int nms(vector<box>& boxes, float threshold)
// {
//     int cur_index = 0;
//     int invalid_index = boxes.size() - 1;
//     int i = 0;
    
//     for (i = 1; i < boxes.size(); i++) {
//         if (boxes[i].score > boxes[0].score) {
//             swap(boxes[0], boxes[i]);
//         }
//     }

//     i = 1;
//     while (i < boxes.size()) {
//         cout << i << ", " << cur_index << "," << invalid_index << endl;
//         if (iou(boxes[cur_index], boxes[i]) >= threshold) {
//             swap(boxes[i], boxes[invalid_index]);
//             invalid_index--;
//         } else {
//             i++;
//         }
//         if (invalid_index == i) {
//             i = ++cur_index + 1;
//         }
//     }

//     return invalid_index;
// }


// every time do nms for the box with max score . 
// this can easily change to the loop style to accerate a bit
void nms_recursive(vector<box>& boxes, int& begin_index, int& end_index, float threshold)
{
    if (begin_index >= end_index ) return; 
    // cout << "nms:" << begin_index << ", " << end_index<< endl; 
    // print_boxes(boxes);

    int i = 0;
    
    for (i = begin_index+1; i <= end_index; i++) {
        if (boxes[i].score > boxes[begin_index].score) {
            swap(boxes[begin_index], boxes[i]);
        }
    }

    i = 1;
    while (begin_index+i <= end_index) {
         
        if (iou(boxes[begin_index], boxes[begin_index+i]) >= threshold) {
            swap(boxes[begin_index+i], boxes[end_index--]);
        } else {
            i++;
        }
    }
    print_boxes(boxes);

    nms_recursive(boxes, ++begin_index , end_index, threshold);
}



int main()
{
    vector<box> boxes= {    
        {100,100,210,210,0.72},
        {230,250,450,450,0.8 },
        {220,220,320,330,0.92},
        {90,80,210,210,0.73},
        {230,240,325,330,0.81},
        {220,230,315,340,0.9 } 
                        };
    
    print_boxes(boxes);
    print_ious(boxes);


    int begin_index = 0;
    int end_index = boxes.size() - 1; 
    nms_recursive(boxes, begin_index, end_index, 0.5);
    cout << "result boxes size :" <<  begin_index<< "," << end_index << endl;

    print_boxes(boxes);
    
    return 0;
}