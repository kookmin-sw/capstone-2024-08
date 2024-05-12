import torch
from pathlib import Path
from sklearn.cluster import KMeans
from model import VAE
from utils import create_dataset
import numpy as np
import  sys, argparse
import configparser


def generate_latent_data(model, dataloader):
    model.eval()
    latent_data = []

    with torch.no_grad():
        for data in dataloader:
            _, mu, _ = model.forward(data.to(device))
            latent_data.append(mu.cpu().detach())

    latent_data = torch.cat(latent_data, dim=0)
    return latent_data.numpy()


# Function to perform K-means clustering
def kmeans_clustering(data, num_clusters):
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    kmeans.fit(data)
    cluster_centers = kmeans.cluster_centers_
    cluster_labels = kmeans.labels_
    return cluster_centers, cluster_labels


if __name__ == "__main__":

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default ='./default.ini' , help='path to the config file')
    args = parser.parse_args()

    # Get configs
    config_path = args.config
    config = configparser.ConfigParser(allow_no_value=True)
    try: 
        config.read(config_path)
    except FileNotFoundError:
        print('Config File Not Found at {}'.format(config_path))
        sys.exit()
    print(config.values())
    sampling_rate = config['audio'].getint('sampling_rate')
    hop_length = config['audio'].getint('hop_length')
    segment_length = config['audio'].getint('segment_length')

    latent_dim = config['VAE'].getint('latent_dim')
    n_units = config['VAE'].getint('n_units')
    n_hidden_units = config['VAE'].getint('n_hidden_units')
    batch_size = config['training'].getint('batch_size')


    num_clusters = 10
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model_checkpoint_path = 'run/002/model/best_model.pt'

    model = VAE(segment_length, n_units, n_hidden_units, latent_dim).to(device)

    model = torch.load(model_checkpoint_path, map_location=torch.device(device))
    model.to(device) 
    model.eval()

    test_dataloader, test_dataset_len = create_dataset("filelists/test.txt", segment_length, sampling_rate, hop_length, batch_size)

    latent_data = generate_latent_data(model, test_dataloader)

    cluster_centers, cluster_labels = kmeans_clustering(latent_data, num_clusters)


    # Print cluster labels and number of data points in each cluster
    for cluster_id in range(num_clusters):
        num_points_in_cluster = sum(cluster_labels == cluster_id)
        print(f"Cluster {cluster_id}: {num_points_in_cluster} data points")

    print("Cluster Centroids:")
    print(cluster_centers)
    print("Cluster Labels:")
    print(cluster_labels)
    
    output_file = model_checkpoint_path / "cluster_results.txt"
    np.savetxt(output_file, cluster_centers)
    with open(output_file, "a") as f:
        f.write("\n")
        np.savetxt(f, cluster_labels)