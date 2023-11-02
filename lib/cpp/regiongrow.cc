#include <algorithm>
#include <iostream>
#include <queue>
#include <unordered_map>
#include <map>
#include <vector>
#include <cstring>
#include <unordered_map>
#include <cmath>

extern "C" {

float distance(float* v1, float* v2, int numC) {
	
	float d = 0.0;
	for (int i = 0; i < numC; ++i) {
		d += (v1[i] - v2[i]) * (v1[i] - v2[i]);
	}
	float result = std::sqrt(d);
	// std::cout << result << std::endl;
	return result;
}

void getdata(float* a){
	std::vector<std::vector<float>> f(2,std::vector<float>(4));
	int k=0;
	for(int i=0;i<2;i++){
		for(int j=0;j<4;j++){
			f[i][j]=a[k];
			k++;
		}
	}
	float result = distance(f[0].data(),f[1].data(),4);
	std::cout << result << std::endl;
}

void RegionGrowing_with_embed(float* embedding, int numC, int* boundary, int* F, int numV, int numF, int* face_labels, int* output_mask, float score_thres, int* output_label) {
	/*
	embedding: 顶点的特征向量，维度为 numV * numC
	numC: 特征向量的维度
	*/
	std::vector<std::vector<float>> point_embed(numV,std::vector<float>(numC));
	for(int i=0;i<numV;i++){
		for(int j=0;j<numC;j++){
			point_embed[i][j]=embedding[i*numC+j];
		}
	}

	std::vector<std::unordered_map<int, int> > v_neighbors(numV);
	for (int i = 0; i < numF; ++i) {
		for (int j = 0; j < 3; ++j) {
			int b = boundary[i * 3 + j];	// 0 or 1，三角面片每个半边的边界预测
			int v0 = F[i * 3 + j];
			int v1 = F[i * 3 + (j + 1) % 3];	// 三角面片的三个顶点索引
			v_neighbors[v0][v1] = b;
			v_neighbors[v1][v0] = b;	// 邻接顶点和边界预测邻接图
		}
	}

	// for (auto& nv : v_neighbors[0]) {
	// 	std::cout << nv.first << std::endl;
	// }

	std::vector<int> mask(numV, -2);	// 初始化每个顶点的聚类标签为 -2
	std::vector<int> boundary_neighbors(numV, 0);	// 记录邻边周围点存在边界点的点，为1则表示当前点邻边存在边界点
	int num_boundary = 0;
	for (int i = 0; i < numV; ++i) {
		int is_boundary = 0;	// 记录当前顶点周围的边界半边数量
		for (auto& info : v_neighbors[i]) {
			if (info.second == 1) {
				is_boundary += 1;
				if (boundary_neighbors[i] == 0) {
					boundary_neighbors[i] = 1;
				}
			}
		}
		if (is_boundary >= v_neighbors[i].size() * score_thres) {
			mask[i] = -1;
			num_boundary += 1;
		}	// 如果当前顶点周围的边界半边数量大于阈值，则将当前顶点标记为边界顶点
	}

	if (output_mask) {	// 如果需要输出边界顶点的 mask
		int b = 0;
		for (int i = 0; i < numV; ++i) {
			output_mask[i] = (mask[i] == -1) ? 1 : 0;	// 将边界顶点的 mask 设置为 1
			b += output_mask[i];
		}
	}

	int num_labels = 0;	// 记录聚类的数量
	for (int i = 0; i < mask.size(); ++i) {
		if (mask[i] == -2) {
			std::queue<int> q;
			q.push(i);
			mask[i] = num_labels;
			while (!q.empty()) {
				int v = q.front();
				q.pop();
				for (auto& nv : v_neighbors[v]) {
					if (nv.second == 0 && mask[nv.first] == -2) {
						if (boundary_neighbors[v] == 1) {	// 如果当前顶点邻边存在边界点，则判断当前顶点与邻边点的特征向量距离，如果小于阈值，则将邻边点标记为当前顶点的聚类
							if (distance(point_embed[v].data(),point_embed[nv.first].data(),numC)<0.1){
								mask[nv.first] = num_labels;
								q.push(nv.first);
							}
						} else {
							mask[nv.first] = num_labels;
							q.push(nv.first);
						}
						// if (distance(point_embed[v].data(),point_embed[nv.first].data(),numC)<0.1){
						// 	mask[nv.first] = num_labels;
						// 	q.push(nv.first);
						// }
						// mask[nv.first] = num_labels;
						// q.push(nv.first);
					}
				}
			}
			num_labels += 1;	// 从当前顶点开始，广度优先搜索，将所有与当前顶点相邻的非边界顶点标记为当前顶点的聚类
		}
	}

	for (int i = 0; i < numF; ++i) {
		int label = -1;
		for (int j = 0; j < 3; ++j) {
			if (mask[F[i * 3 + j]] >= 0) {
				label = mask[F[i * 3 + j]];
				break;
			}
		}
		face_labels[i] = label;
	}

	for (int i = 0; i < numV; ++i) {
		output_label[i] = mask[i];
	}
}

void RegionGrowing(int* boundary, int* F, int numV, int numF, int* face_labels, int* output_mask, float score_thres, int* output_label) {
	std::vector<std::unordered_map<int, int> > v_neighbors(numV);
	for (int i = 0; i < numF; ++i) {
		for (int j = 0; j < 3; ++j) {
			int b = boundary[i * 3 + j];	// 0 or 1，三角面片每个半边的边界预测
			int v0 = F[i * 3 + j];
			int v1 = F[i * 3 + (j + 1) % 3];	// 三角面片的三个顶点索引
			v_neighbors[v0][v1] = b;
			v_neighbors[v1][v0] = b;	// 邻接顶点和边界预测邻接图
		}
	}

	// for (auto& nv : v_neighbors[0]) {
	// 	std::cout << nv.first << std::endl;
	// }

	std::vector<int> mask(numV, -2);	// 初始化每个顶点的聚类标签为 -2
	int num_boundary = 0;
	for (int i = 0; i < numV; ++i) {
		int is_boundary = 0;	// 记录当前顶点周围的边界半边数量
		for (auto& info : v_neighbors[i]) {
			if (info.second == 1) {
				is_boundary += 1;
			}
		}
		if (is_boundary >= v_neighbors[i].size() * score_thres) {
			mask[i] = -1;
			num_boundary += 1;
		}	// 如果当前顶点周围的边界半边数量大于阈值，则将当前顶点标记为边界顶点
	}

	if (output_mask) {	// 如果需要输出边界顶点的 mask
		int b = 0;
		for (int i = 0; i < numV; ++i) {
			output_mask[i] = (mask[i] == -1) ? 1 : 0;	// 将边界顶点的 mask 设置为 1
			b += output_mask[i];
		}
	}

	int num_labels = 0;	// 记录聚类的数量
	for (int i = 0; i < mask.size(); ++i) {
		if (mask[i] == -2) {
			std::queue<int> q;
			q.push(i);
			mask[i] = num_labels;
			while (!q.empty()) {
				int v = q.front();
				q.pop();
				for (auto& nv : v_neighbors[v]) {
					if (nv.second == 0 && mask[nv.first] == -2) {
						mask[nv.first] = num_labels;
						q.push(nv.first);
					}
				}
			}
			num_labels += 1;	// 从当前顶点开始，广度优先搜索，将所有与当前顶点相邻的非边界顶点标记为当前顶点的聚类
		}
	}

	for (int i = 0; i < numF; ++i) {
		int label = -1;
		for (int j = 0; j < 3; ++j) {
			if (mask[F[i * 3 + j]] >= 0) {
				label = mask[F[i * 3 + j]];
				break;
			}
		}
		face_labels[i] = label;
	}

	for (int i = 0; i < numV; ++i) {
		output_label[i] = mask[i];
	}
}

void RegionGrowing_Point(int* bb, int* boundary, int* F, int numV, int numF, int* face_labels, int* output_mask, float score_thres) {
	std::vector<std::unordered_map<int, int> > v_neighbors(numV);
	for (int i = 0; i < numF; ++i) {
		for (int j = 0; j < 3; ++j) {
			int b = boundary[i * 3 + j];	// 0 or 1，三角面片每个半边的边界预测
			int v0 = F[i * 3 + j];
			int v1 = F[i * 3 + (j + 1) % 3];	// 三角面片的三个顶点索引
			v_neighbors[v0][v1] = b;
			v_neighbors[v1][v0] = b;	// 邻接顶点和边界预测邻接图
		}
	}

	std::vector<int> mask(numV, -2);	// 初始化每个顶点的聚类标签为 -2
	// int num_boundary = 0;
	// for (int i = 0; i < numV; ++i) {
	// 	int is_boundary = 0;	// 记录当前顶点周围的边界半边数量
	// 	for (auto& info : v_neighbors[i]) {
	// 		if (info.second == 1) {
	// 			is_boundary += 1;
	// 		}
	// 	}
	// 	if (is_boundary >= v_neighbors[i].size() * score_thres) {
	// 		mask[i] = -1;
	// 		num_boundary += 1;
	// 	}	// 如果当前顶点周围的边界半边数量大于阈值，则将当前顶点标记为边界顶点
	// }
	for (int i = 0; i < numV; ++i) {
		if (bb[i] == 1) {
			mask[i] = -1;
		}
	}

	if (output_mask) {	// 如果需要输出边界顶点的 mask
		int b = 0;
		for (int i = 0; i < numV; ++i) {
			output_mask[i] = (mask[i] == -1) ? 1 : 0;	// 将边界顶点的 mask 设置为 1
			b += output_mask[i];
		}
	}

	int num_labels = 0;	// 记录聚类的数量
	for (int i = 0; i < mask.size(); ++i) {
		if (mask[i] == -2) {
			std::queue<int> q;
			q.push(i);
			mask[i] = num_labels;
			while (!q.empty()) {
				int v = q.front();
				q.pop();
				for (auto& nv : v_neighbors[v]) {
					if (mask[nv.first] == -2) {
						mask[nv.first] = num_labels;
						q.push(nv.first);
					}
				}
			}
			num_labels += 1;	// 从当前顶点开始，广度优先搜索，将所有与当前顶点相邻的非边界顶点标记为当前顶点的聚类
		}
	}

	for (int i = 0; i < numF; ++i) {
		int label = -1;
		for (int j = 0; j < 3; ++j) {
			if (mask[F[i * 3 + j]] >= 0) {
				label = mask[F[i * 3 + j]];
				break;
			}
		}
		face_labels[i] = label;
	}
}

int GetParent(std::vector<int>& parent, int j) {
	if (j == parent[j])
		return j;
	parent[j] = GetParent(parent, parent[j]);
	return parent[j];
}

void RegionGrowingMerge(float* boundary, int* F, int numV, int numF, int* face_labels, int* mask, float score_thres) {
	std::map<std::pair<int, int>, std::vector<int> > edge_to_faces;
	std::map<std::pair<int, int>, float> edge_scores;
	std::vector<float> face_scores(numF, 0);
	for (int i = 0; i < numF; ++i) {
		for (int j = 0; j < 3; ++j) {
			int v0 = F[i * 3 + j];
			int v1 = F[i * 3 + (j + 1) % 3];
			if (v0 > v1)
				std::swap(v0, v1);
			float score = boundary[j * numF + i];
			auto k = std::make_pair(v0, v1);
			edge_scores[k] = score;
			if (edge_to_faces.count(k)) {
				edge_to_faces[k].push_back(i);
			} else {
				std::vector<int> faces;
				faces.push_back(i);
				edge_to_faces[k] = faces;
			}
			face_scores[i] += score;
		}
	}

	std::vector<int> parents(numF), visited(numF, 0);
	for (int i = 0; i < parents.size(); ++i) {
		parents[i] = i;
	}

	std::map<std::pair<int, int>, float > face_edge_scores;
	for (auto& p : edge_to_faces) {
		auto e = p.first;
		float k = edge_scores[e] * 2;
		for (auto& f1 : p.second) {
			for (auto& f2 : p.second) {
				if (f1 < f2) {
					auto k = std::make_pair(f1, f2);
					float edge = edge_scores[e];
					float s = face_scores[f1] + face_scores[f2] - edge * 2;
					face_edge_scores[k] = s;
				}
			}
		}
	}
	for (float thres = 0; thres <= score_thres; thres += 0.1) {
		bool update = true;
		while (update) {
			update = false;
			std::map<std::pair<int, int>, std::pair<float, int> > merged_face_edge_scores;
			std::vector<std::pair<int, int> > removed_keys;
			for (auto& p : face_edge_scores) {
				auto e = p.first;
				int v1 = GetParent(parents, e.first);
				int v2 = GetParent(parents, e.second);
				if (v1 == v2) {
					removed_keys.push_back(e);
					continue;
				}
				if (v1 > v2) {
					std::swap(v1, v2);
				}
				auto it = merged_face_edge_scores.find(std::make_pair(v1, v2));
				if (it != merged_face_edge_scores.end()) {
					it->second.first += p.second;
					it->second.second += 1;
				} else {
					merged_face_edge_scores[std::make_pair(v1, v2)] = std::make_pair(p.second, 1);
				}
			}
			for (auto& k : removed_keys)
				face_edge_scores.erase(k);

			memset(visited.data(), 0, sizeof(int) * numF);
			std::vector<std::pair<float, std::pair<int, int> > > edge_set;
			for (auto& info : merged_face_edge_scores) {
				float s = info.second.first / info.second.second;
				edge_set.push_back(std::make_pair(s, info.first));
			}
			std::sort(edge_set.begin(), edge_set.end());
			for (auto& e : edge_set) {
				if (e.first > thres)
					break;
				int v1 = e.second.first;
				int v2 = e.second.second;
				if (visited[v1] || visited[v2]) {
					continue;
				}
				parents[v2] = v1;
				visited[v1] = 1;
				visited[v2] = 1;
				update = true;
			}
		}
	}
	int numgroup = 0;
	for (auto& v : visited)
		v = -1;
	for (int i = 0; i < visited.size(); ++i) {
		int p = GetParent(parents, i);
		if (visited[p] == -1) {
			visited[p] = numgroup++;
		}
		face_labels[i] = visited[p];
	}

	std::vector<int> group_count(numgroup, 0);
	for (int i = 0; i < visited.size(); ++i) {
		group_count[face_labels[i]] += 1;
	}
	std::vector<int> group_remap(numgroup, 0);
	numgroup = 0;
	for (int i = 0; i < group_remap.size(); ++i) {
		if (group_count[i] < 5) {
			group_remap[i] = -100;
		} else {
			group_remap[i] = numgroup++;
		}
	}
	for (int i = 0; i < visited.size(); ++i) {
		face_labels[i] = group_remap[face_labels[i]];
	}
}

void BoundaryRegionGrowing(int* boundary, int* F, int numV, int numF, int* face_labels, int* output_mask) {
	std::vector<std::unordered_map<int, int> > v_neighbors(numV);
	for (int i = 0; i < numF; ++i) {
		for (int j = 0; j < 3; ++j) {
			int b = boundary[j * numF + i];
			int v0 = F[i * 3 + j];
			int v1 = F[i * 3 + (j + 1) % 3];
			v_neighbors[v0][v1] = b;
			v_neighbors[v1][v0] = b;
		}
	}

	std::vector<int> mask(numV, -2);
	int num_boundary = 0;
	for (int i = 0; i < numV; ++i) {
		if (boundary[i]) {
			num_boundary += 1;
			mask[i] = -1;
		}
	}
	printf("Num boundary %d of %d\n", num_boundary, numV);

	if (output_mask) {
		int b = 0;
		for (int i = 0; i < numV; ++i) {
			output_mask[i] = (mask[i] == -1) ? 1 : 0;
			b += output_mask[i];
		}
		printf("Num boundary %d\n", b);
	}

	int num_labels = 0;
	for (int i = 0; i < mask.size(); ++i) {
		if (mask[i] == -2) {
			std::queue<int> q;
			q.push(i);
			mask[i] = num_labels;
			while (!q.empty()) {
				int v = q.front();
				q.pop();
				for (auto& nv : v_neighbors[v]) {
					if (nv.second == 0 && mask[nv.first] == -2) {
						mask[nv.first] = num_labels;
						q.push(nv.first);
					}
				}
			}
			num_labels += 1;
		}
	}

	printf("Num label %d\n", num_labels);
	for (int i = 0; i < numF; ++i) {
		int label = -1;
		for (int j = 0; j < 3; ++j) {
			if (mask[F[i * 3 + j]] >= 0) {
				label = mask[F[i * 3 + j]];
				break;
			}
		}
		face_labels[i] = label;
	}
}
};
