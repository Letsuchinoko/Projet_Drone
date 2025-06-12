package com.example.test

import android.bluetooth.BluetoothDevice
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.TextView
import androidx.recyclerview.widget.RecyclerView

class BluetoothDeviceAdapter(private val devices: MutableList<BluetoothDevice>) :
    RecyclerView.Adapter<BluetoothDeviceAdapter.BluetoothDeviceViewHolder>() {

    class BluetoothDeviceViewHolder(itemView: View) : RecyclerView.ViewHolder(itemView) {
        val deviceName: TextView = itemView.findViewById(R.id.textViewDeviceName)
        val deviceAddress: TextView = itemView.findViewById(R.id.textViewDeviceAddress)
    }

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): BluetoothDeviceViewHolder {
        val view = LayoutInflater.from(parent.context)
            .inflate(R.layout.device_item, parent, false)
        return BluetoothDeviceViewHolder(view)
    }

    override fun onBindViewHolder(holder: BluetoothDeviceViewHolder, position: Int) {
        val device = devices[position]
        holder.deviceName.text = device.name ?: "Unknown Device"
        holder.deviceAddress.text = device.address
    }

    override fun getItemCount(): Int = devices.size

    fun addDevice(device: BluetoothDevice) {
        if (!devices.contains(device)) {
            devices.add(device)
            notifyItemInserted(devices.size - 1)
        }
    }

    fun clearDevices() {
        devices.clear()
        notifyDataSetChanged()
    }
} 