//
// Copyright 2026 Rupert Nash, EPCC, University of Edinburgh
//
#include "h5io.h"

#include <array>
#include <concepts>
#include <span>
#include <vector>

#include <hdf5.h>
#include "ufield.h"
#include "util.h"

namespace h5 {
    template <typename T>
    struct type_traits;

#define DECLARE_H5T(tn, ht) \
template<> struct type_traits<tn> {\
    static hid_t get() { return ht; } \
}

    DECLARE_H5T(int, H5T_NATIVE_INT);
    DECLARE_H5T(unsigned, H5T_NATIVE_UINT);
    DECLARE_H5T(unsigned long, H5T_NATIVE_ULONG);
    DECLARE_H5T(unsigned long long, H5T_NATIVE_ULLONG);
    DECLARE_H5T(double, H5T_NATIVE_DOUBLE);
    DECLARE_H5T(long long, H5T_NATIVE_LLONG);

    class error : public std::runtime_error {
        using std::runtime_error::runtime_error;
    };

#define STRINGIFY(x) STRINGIFY2(x)
#define STRINGIFY2(x) #x
#define LINE_STRING STRINGIFY(__LINE__) \
                                        \
// Wrap a call that returns an object ID, throwing on error
#define CHECK_ID(func, ...) [&](){ \
    hid_t res = func(__VA_ARGS__); \
    if (res < 0) \
        throw ::h5::error("HDF5 error in " # func " at " __FILE__ ":" LINE_STRING); \
    return res; \
}()
// Wrap a call that returns an error code, throwing on error
#define CHECK_ERR(func, ...) [&]() { \
    herr_t res = func(__VA_ARGS__); \
    if (res < 0) \
        throw ::h5::error("HDF5 error in " # func " at " __FILE__ ":" LINE_STRING); \
}()
// Convert a call that returns a trinary (true, false, error) into a binary (true/false) or exception (error)
#define CHECK_TRI(func, ...) [&]() -> bool { \
    htri_t res = func(__VA_ARGS__); \
    if (res < 0) \
        throw ::h5::error("HDF5 error in " # func " at " __FILE__ ":" LINE_STRING); \
    return res > 0; \
}()

    // Delightfully, HDF5 uses unsigned long long which is NOT the same
    // as std::size_t, despite both being 64 bit unsigned integers. Ah,
    // C++.
    template <std::size_t N>
    using SA = std::array<hsize_t, N>;

    class object {
    protected:
        hid_t id;

    public:
        object(hid_t i) : id(i) {
        }

        object() : id(H5I_INVALID_HID) {
        }

        object(object const&) = delete;

        object& operator=(object const&) = delete;

        virtual ~object() = default;

        operator hid_t() const {
            return id;
        }
    };

    class pl : public object {
    public:
        using object::object;

        template <hsize_t...>
        friend
        class chunk;

        pl(pl&& src) noexcept: object(src.id) {
            src.id = H5I_INVALID_HID;
        }

        pl& operator=(pl&& src)  noexcept {
            if (&src != this) {
                close();
                std::swap(id, src.id);
            }
            return *this;
        }

        ~pl() override {
            close();
        }

        void close() {
            if (id != H5I_INVALID_HID) {
                CHECK_ERR(H5Pclose, id);
                id = H5I_INVALID_HID;
            }
        }

        operator hid_t() const {
            return id == H5I_INVALID_HID ? H5P_DEFAULT : id;
        }
    };

    struct attr {
        hid_t parent;
        char const* name;

        void operator=(char const* s) {
            hid_t space = CHECK_ID(H5Screate, (H5S_SCALAR));
            hid_t memtype = CHECK_ID(H5Tcopy, (H5T_C_S1));
            H5Tset_size(memtype, H5T_VARIABLE);
            hid_t a = CHECK_ID(H5Acreate, parent, name, memtype, space, H5P_DEFAULT, H5P_DEFAULT);
            CHECK_ERR(H5Awrite, a, memtype, &s);
            H5Tclose(memtype);
            H5Sclose(space);
            H5Aclose(a);
        }

        template <typename T>
        void operator=(T const& val) {
            hid_t space = CHECK_ID(H5Screate, (H5S_SCALAR));
            hid_t type = type_traits<T>::get();
            hid_t a;
            if (CHECK_TRI(H5Aexists, parent, name)) {
                a = CHECK_ID(H5Aopen, parent, name, H5P_DEFAULT);
                // Should check stuff but hopefully the write will error!
            } else {
                a = CHECK_ID(H5Acreate, parent, name, type, space, H5P_DEFAULT, H5P_DEFAULT);
            }
            CHECK_ERR(H5Awrite, a, type, &val);
            H5Sclose(space);
            H5Aclose(a);
        }

        template <typename T>
        void operator=(std::initializer_list<T> is) {
            // Bit naughty to cast but the span is only used through the const ref
            *this = std::span<T>((T*) data(is), is.size());
        }

        template <typename T, std::size_t N>
        void operator=(std::span<T, N> const& data) {
            using U = std::decay_t<T>;
            hsize_t size = data.size();
            hid_t space = CHECK_ID(H5Screate_simple, 1, &size, &size);
            hid_t type = type_traits<U>::get();
            hid_t a = CHECK_ID(H5Acreate, parent, name, type, space, H5P_DEFAULT, H5P_DEFAULT);
            CHECK_ERR(H5Awrite, a, type, data.data());
            H5Sclose(space);
            H5Aclose(a);
        }

        template <typename T>
        operator T() {
            hid_t a = CHECK_ID(H5Aopen, parent, name, H5P_DEFAULT);
            hid_t space = CHECK_ID(H5Aget_space, a);
            if (H5Sget_simple_extent_type(space) != H5S_SCALAR)
                throw std::runtime_error("Can't read a non scalar by cast");
            T ans;
            CHECK_ERR(H5Aread, a, type_traits<T>::get(), &ans);
            H5Sclose(space);
            H5Aclose(a);
            return ans;
        }

        template <typename T, std::size_t N>
        void read(std::span<T, N> dest) {
            hid_t a = CHECK_ID(H5Aopen, parent, name, H5P_DEFAULT);
            hid_t space = CHECK_ID(H5Aget_space, a);
            if (H5Sget_simple_extent_ndims(space) != 1)
                throw std::runtime_error("Must be 1D to get into span");
            hsize_t file_count;
            H5Sget_simple_extent_dims(space, &file_count, nullptr);
            if (file_count != dest.size())
                throw std::runtime_error("File size and requested size don't match");
            CHECK_ERR(H5Aread, a, type_traits<T>::get(), dest.data());
            H5Sclose(space);
            H5Aclose(a);
        }
    };

    struct attr_holder {
        hid_t id;

        attr operator[](char const* name) {
            return {id, name};
        }
    };

    class dataspace : public object {
    public:
        using object::object;

        static dataspace Scalar() {
            return CHECK_ID(H5Screate, H5S_SCALAR);
        }

        template <std::size_t RANK>
        static dataspace create_simple(SA<RANK> const& shape) {
            return {CHECK_ID(H5Screate_simple, RANK, shape.data(), nullptr)};
        }

        template <std::integral... Ints>
        static dataspace create_simple(Ints... shape) {
            return create_simple<sizeof...(Ints)>({shape...});
        }

        template <unsigned rank>
        static dataspace create_resizable(SA<rank> shape, SA<rank> max) {
            return {CHECK_ID(H5Screate_simple, rank, shape.data(), max.data())};
        }

        dataspace(dataspace&& src) noexcept: object(src.id) {
            src.id = H5I_INVALID_HID;
        }

        dataspace& operator=(dataspace&& src)  noexcept {
            if (&src != this) {
                close();
                std::swap(id, src.id);
            }
            return *this;
        }

        ~dataspace() override {
            close();
        }

        void close() {
            if (id != H5I_INVALID_HID) {
                H5Sclose(id);
                id = H5I_INVALID_HID;
            }
        }
    };

    class dataset : public object {
    public:
        using object::object;

        dataset(dataset&& src) noexcept: object(src.id) {
            src.id = H5I_INVALID_HID;
        }

        dataset& operator=(dataset&& src)  noexcept {
            if (&src != this) {
                close();
                std::swap(id, src.id);
            }
            return *this;
        }

        ~dataset() override {
            close();
        }

        void close() {
            if (id != H5I_INVALID_HID) {
                H5Dclose(id);
                id = H5I_INVALID_HID;
            }
        }

        [[nodiscard]] dataspace get_space() const {
            return CHECK_ID(H5Dget_space, id);
        }
    };

    struct chunk_base {
        [[nodiscard]] virtual pl dcpl() const = 0;
    };

    template <hsize_t... Dims>
    struct chunk : public chunk_base {
        static constexpr hsize_t rank = sizeof...(Dims);
        // Use index/size_t rather than the more natural hsize because easier to bodge (below)
        static constexpr nd::index<rank> shape = {Dims...};

        [[nodiscard]] pl dcpl() const override {
            // Going to cheat with a cast (as all 64 bit unsigned ints
            // are the same to hardware, just not the C++ VM)
            static_assert(sizeof(hsize_t) == sizeof(std::size_t));
            auto dcpl_id = H5Pcreate(H5P_DATASET_CREATE);
            CHECK_ERR(H5Pset_chunk, dcpl_id, rank, (hsize_t const*)shape.data());
            return {dcpl_id};
        }
    };

    class group : public object {
        friend class file;

    public:
        using object::object;

        group(group&& src) noexcept: object(src.id) {
            src.id = H5I_INVALID_HID;
        }

        group& operator=(group&& src) {
            if (&src != this) {
                close();
                std::swap(id, src.id);
            }
            return *this;
        }

        ~group() override {
            close();
        }

        void close() {
            if (id != H5I_INVALID_HID) {
                H5Gclose(id);
                id = H5I_INVALID_HID;
            }
        }

        attr_holder attrs() {
            return {id};
        }

        group create_group(char const* name) {
            return {CHECK_ID(H5Gcreate, id, name, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT)};
        }

        group open_group(char const* name) {
            return CHECK_ID(H5Gopen, id, name, H5P_DEFAULT);
        }

        template <typename T>
        dataset create_dataset(char const* name, dataspace const& space, hid_t dcpl) {
            hid_t type = type_traits<T>::get();
            return CHECK_ID(H5Dcreate, id, name, type, space,
                            H5P_DEFAULT, dcpl == H5I_INVALID_HID ? H5P_DEFAULT : dcpl, H5P_DEFAULT);
        }

        template <typename T, std::integral... Ints>
        dataset create_dataset(char const* name, Ints... shape) {
            auto space = dataspace::create_simple(shape...);
            return create_dataset < T > (name, space);
        }

        dataset open_dataset(char const* name) {
            return CHECK_ID(H5Dopen, id, name, H5P_DEFAULT);
        }
    };

    class file : public object {
    public:
        static file create(char const* fn, unsigned flags) {
            return {CHECK_ID(H5Fcreate, fn, flags, H5P_DEFAULT, H5P_DEFAULT)};
        }
        static file open(char const* fn, unsigned flags) {
            return {CHECK_ID(H5Fopen, fn, flags, H5P_DEFAULT)};
        }
        using object::object;

        file(file&& src) noexcept: object(src.id) {
            src.id = H5I_INVALID_HID;
        }

        file& operator=(file&& src)  noexcept {
            if (&src != this) {
                close();
                std::swap(id, src.id);
            }
            return *this;
        }

        void close() {
            if (id != H5I_INVALID_HID) {
                H5Fclose(id);
                id = H5I_INVALID_HID;
            }
        }

        ~file() override {
            close();
        }

        group create_group(char const* name) {
            return {CHECK_ID(H5Gcreate, id, name, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT)};
        }

        group open_group(char const* name) {
            return CHECK_ID(H5Gopen, id, name, H5P_DEFAULT);
        }
    };

    template <typename T>
    void append1d(dataset& ds, T const& value) {
        auto space = ds.get_space();
        if (H5Sget_simple_extent_ndims(space) != 1)
            throw std::runtime_error("Invalid rank");
        hsize_t N;
        H5Sget_simple_extent_dims(space, &N, nullptr);
        auto Np1 = N + 1;
        CHECK_ERR(H5Dextend, ds, &Np1);
        auto memspace = dataspace::Scalar();
        auto filespace = ds.get_space();
        hsize_t count = 1;
        CHECK_ERR(H5Sselect_hyperslab, filespace, H5S_SELECT_SET, &N, nullptr, &count, nullptr);
        CHECK_ERR(H5Dwrite, ds, type_traits<T>::get(), memspace, filespace, H5P_DEFAULT, &value);
    }

    struct Impl {
        // File
        file f;
        group steps;
        dataset values;
        std::vector<dataset> static_pdos;
        std::vector<dataset> changing_pdos;
        dataset u_now;
        dataset u_prev;
        shape_t data_shape;
        using CHUNK = h5::chunk<1, 8, 8, 8>;
    };
}

using SA4 = h5::SA<4>;

void chunk_transpose_write(
        h5::dataset& dset, hsize_t start_t,
        array3d const& data, nd::index<3> data_start, nd::index<3> data_count
) {
    using CHUNK = h5::Impl::CHUNK;
    nd::array<double, 4> chunk(h5::Impl::CHUNK::shape);
    // structured binding cannot be constexpr *sigh*
    constexpr auto CZ = CHUNK::shape[1];
    constexpr auto CY = CHUNK::shape[2];
    constexpr auto CX = CHUNK::shape[3];
    auto [dx, dy, dz] = data_count;
    auto ncx = ceildiv(dx, CX);
    auto ncy = ceildiv(dy, CY);
    auto ncz = ceildiv(dz, CZ);

    SA4 chunk_start_pos = {start_t, 0, 0, 0};
    for (auto ci = 0U; ci < ncx; ++ci) {
        chunk_start_pos[3] = ci*CX;
        for (auto cj = 0U; cj < ncy; ++cj) {
            chunk_start_pos[2] = cj*CY;
            for (auto ck = 0U; ck < ncz; ++ck) {
                chunk_start_pos[1] = ck*CZ;
                // Fill the chunk!
                std::fill_n(chunk.data(), chunk.size(), 0);
                for (auto i = 0U; i < CX; ++i) {
                    auto di = data_start[0] + ci*CX + i;
                    if (di >= dx) continue;
                    for (auto j = 0U; j < CY; ++j) {
                        auto dj = data_start[1] + cj*CY + j;
                        if (dj >= dy) continue;
                        for (auto k = 0U; k < CZ; ++k) {
                            auto dk = data_start[2] + ck*CZ + k;
                            if (dk >= dz) continue;
                            // Copy one value
                            chunk(0U, k, j, i) = data(di, dj, dk);
                        }
                    }
                }
                // Write the chunk directly
                CHECK_ERR(H5Dwrite_chunk, dset, H5P_DEFAULT, 0, chunk_start_pos.data(), chunk.size()*sizeof(double), chunk.data());
            }
        }
    }
}
void chunk_transpose_write(h5::dataset& dset, hsize_t start_t, array3d const& data) {
    chunk_transpose_write(dset, start_t, data, {0, 0, 0}, data.shape());
}

void chunk_transpose_read(
        h5::dataset& dset, hsize_t start_t,
        array3d& data, nd::index<3> data_start, nd::index<3> data_count
) {
    using CHUNK = h5::Impl::CHUNK;
    nd::array<double, 4> chunk(h5::Impl::CHUNK::shape);
    // structured binding cannot be constexpr *sigh*
    constexpr auto CZ = CHUNK::shape[1];
    constexpr auto CY = CHUNK::shape[2];
    constexpr auto CX = CHUNK::shape[3];
    auto [dx, dy, dz] = data_count;
    auto ncx = ceildiv(dx, CX);
    auto ncy = ceildiv(dy, CY);
    auto ncz = ceildiv(dz, CZ);

    SA4 chunk_start_pos = {start_t, 0, 0, 0};
    for (auto ci = 0U; ci < ncx; ++ci) {
        chunk_start_pos[3] = ci * CX;
        for (auto cj = 0U; cj < ncy; ++cj) {
            chunk_start_pos[2] = cj * CY;
            for (auto ck = 0U; ck < ncz; ++ck) {
                chunk_start_pos[1] = ck * CZ;
                // Read the chunk
                std::uint32_t filters;
                CHECK_ERR(H5Dread_chunk, dset, H5P_DEFAULT, chunk_start_pos.data(), &filters, chunk.data());
                for (auto i = 0U; i < CX; ++i) {
                    auto di = data_start[0] + ci * CX + i;
                    if (di >= dx) continue;
                    for (auto j = 0U; j < CY; ++j) {
                        auto dj = data_start[1] + cj * CY + j;
                        if (dj >= dy) continue;
                        for (auto k = 0U; k < CZ; ++k) {
                            auto dk = data_start[2] + ck * CZ + k;
                            if (dk >= dz) continue;
                            // Copy one value
                            data(di, dj, dk) = chunk(0U, k, j, i);
                        }
                    }
                }
            }
        }
    }
}
void chunk_transpose_read(h5::dataset& dset, hsize_t start_t, array3d& data) {
    chunk_transpose_read(dset, start_t, data, {0, 0, 0}, data.shape());
}

// Now we have the definition of h5::Impl, we can default these.
H5IO::H5IO() = default;
H5IO::H5IO(H5IO&&) noexcept = default;
H5IO& H5IO::operator=(H5IO&&) noexcept = default;
H5IO::~H5IO() = default;


H5IO H5IO::from_file(const fs::path& path) {
    H5IO ans;
    ans.m_impl = std::make_unique<h5::Impl>();
    auto& i = *ans.m_impl;
    i.f = h5::file::open(path.c_str(), H5F_ACC_RDWR);
    auto p = ans.get_params();
    i.data_shape = p.shape;
    auto root = i.f.open_group("VTKHDF");
    i.steps = root.open_group("Steps");
    i.values = i.steps.open_dataset("Values");
    auto steps_pdo = i.steps.open_group("PointDataOffsets");
    // Static ones (i.e. will always append a zero here)
    i.static_pdos.emplace_back(steps_pdo.open_dataset("damp"));
    i.static_pdos.emplace_back(steps_pdo.open_dataset("sos"));
    // Changing ones!
    i.changing_pdos.emplace_back(steps_pdo.open_dataset("u"));
    i.changing_pdos.emplace_back(steps_pdo.open_dataset("u_prev"));
    auto pd = root.open_group("PointData");
    i.u_now = pd.open_dataset("u");
    i.u_prev = pd.open_dataset("u_prev");
    return ans;
}

H5IO H5IO::from_params(fs::path const& path, Params const& params) {
    H5IO ans;
    ans.m_impl = std::make_unique<h5::Impl>();
    auto& i = *ans.m_impl;
    i.data_shape = params.shape;
    i.f = h5::file::create(path.c_str(),H5F_ACC_TRUNC);
    auto [nx, ny, nz] = params.shape;
    auto root = i.f.create_group("VTKHDF");
    {
        auto r = root.attrs();
        r["Version"] = {2, 3};
        r["Type"] = "ImageData";
        r["Origin"] = {0.0, 0.0, 0.0};
        r["Spacing"] = {params.dx, params.dx, params.dx};
	// Force it to the right type
        r["WholeExtent"] = std::initializer_list<std::size_t>{0, nx-1, 0, ny-1, 0, nz-1};
        r["Direction"] = {1, 0, 0,
                          0, 1, 0,
                          0, 0, 1};
    }
    // Time step info
    i.steps = root.create_group("Steps");
    i.steps.attrs()["NSteps"] = 0LL;
    auto space = h5::dataspace::create_resizable<1>({0}, {H5S_UNLIMITED});
    // We must chunk the resizable datasets
    auto field_chunk_pl = h5::Impl::CHUNK().dcpl();
    auto offset_chunk_pl = h5::chunk<32>().dcpl();
    // Actual times
    i.values = i.steps.create_dataset<long long>("Values", space, offset_chunk_pl);
    // Offsets into data arrays for given step
    auto steps_pdo = i.steps.create_group("PointDataOffsets");
    // Static ones (i.e. will always append a zero here)
    i.static_pdos.emplace_back(steps_pdo.create_dataset<long long>("damp", space, offset_chunk_pl));
    i.static_pdos.emplace_back(steps_pdo.create_dataset<long long>("sos", space, offset_chunk_pl));
    // Changing ones!
    i.changing_pdos.emplace_back(steps_pdo.create_dataset<long long>("u", space, offset_chunk_pl));
    i.changing_pdos.emplace_back(steps_pdo.create_dataset<long long>("u_prev", space, offset_chunk_pl));

    // Actual data arrays now
    auto pd = root.create_group("PointData");
    // Have to order things as (time, z, y, z) cos VTK *sigh*
    {
        // static fields have ntime = 1 and we won't touch these again
        auto file_space = h5::dataspace::create_simple(1U, nz, ny, nx);
        auto sos_ds = pd.create_dataset<double>("sos", file_space, field_chunk_pl);
        auto damp_ds = pd.create_dataset<double>("damp", file_space, field_chunk_pl);
    }
    {
        // growing fields start empty in time
        auto file_space = h5::dataspace::create_resizable<4>({0u, nz, ny, nx}, {H5S_UNLIMITED, nz, ny, nx});
        i.u_now = pd.create_dataset<double>("u", file_space, field_chunk_pl);
        i.u_prev = pd.create_dataset<double>("u_prev", file_space, field_chunk_pl);
    }
    return ans;
}

void H5IO::put_params(const Params& params) {
    auto wave_params = m_impl->f.create_group("wave_params");
    auto a = wave_params.attrs();
    a["dt"] = params.dt;
    a["dx"] = params.dx;
    a["nBoundaryLayers"] = params.nBoundaryLayers;
    a["nsteps"] = params.nsteps;
    a["out_period"] = params.out_period;
    a["shape"] = std::span(params.shape);
}

Params H5IO::get_params() {
    Params params;
    auto wave_params = m_impl->f.open_group("wave_params");
    auto a = wave_params.attrs();
    params.dt = a["dt"];
    params.dx = a["dx"];
    params.nBoundaryLayers = a["nBoundaryLayers"];
    params.nsteps = a["nsteps"];
    params.out_period= a["out_period"];
    a["shape"].read(std::span(params.shape));
    return params;
}

void H5IO::put_damp(const array3d& damp) {
    auto root = m_impl->f.open_group("VTKHDF");
    auto pd = root.open_group("PointData");
    auto ds = pd.open_dataset("damp");
    chunk_transpose_write(ds, 0, damp);
}

array3d H5IO::get_damp() {
    auto root = m_impl->f.open_group("VTKHDF");
    auto pd = root.open_group("PointData");
    auto ds = pd.open_dataset("damp");
    array3d ans(m_impl->data_shape);
    chunk_transpose_read(ds, 0, ans);
    return ans;
}

void H5IO::put_sos(const array3d& sos) {
    auto root = m_impl->f.open_group("VTKHDF");
    auto pd = root.open_group("PointData");
    auto ds = pd.open_dataset("sos");
    chunk_transpose_write(ds, 0, sos);
}

array3d H5IO::get_sos() {
    auto root = m_impl->f.open_group("VTKHDF");
    auto pd = root.open_group("PointData");
    auto ds = pd.open_dataset("sos");
    array3d ans(m_impl->data_shape);
    chunk_transpose_read(ds, 0, ans);
    return ans;
}

void H5IO::append_u(uField const& u) {
    auto& i = *m_impl;
    int N = i.steps.attrs()["NSteps"];
    // Actual time value
    h5::append1d(i.values, u.time());

    // Static data offsets
    for (auto& pdo: i.static_pdos) {
        h5::append1d(pdo, 0);
    }
    // dynamic
    for (auto& pdo: i.changing_pdos) {
        h5::append1d(pdo, N);
    }
    // Data
    auto extend = [&](h5::dataset& ds) {
        auto space = ds.get_space();
        if (H5Sget_simple_extent_ndims(space) != 4)
            throw std::runtime_error("Invalid rank");
        SA4 shape;
        H5Sget_simple_extent_dims(space, shape.data(), nullptr);
        shape[0] += 1;
        CHECK_ERR(H5Dextend, ds, shape.data());
    };
    extend(i.u_now);
    chunk_transpose_write(i.u_now, N, u.now(), {1,1,1}, i.data_shape);
    extend(i.u_prev);
    chunk_transpose_write(i.u_prev, N, u.prev(), {1,1,1}, i.data_shape);
    // update count
    i.steps.attrs()["NSteps"] = N + 1;
}

uField H5IO::get_last_u() {
    auto& i = *m_impl;

    auto space = i.values.get_space();
    hsize_t N;
    H5Sget_simple_extent_dims(space, &N, nullptr);
    std::vector<long long> times(N);
    CHECK_ERR(H5Dread, i.values, h5::type_traits<long long>::get(), space, space, H5P_DEFAULT, times.data());

    auto const& shape = i.data_shape;
    auto padded_shape = shape;
    for (auto& s: padded_shape)
        s += 2;

    auto last = N-1;
    uField u(array3d{padded_shape}, times[last]);
    chunk_transpose_read(i.u_now, last, u.now(), {1,1,1}, shape);
    chunk_transpose_read(i.u_prev, last, u.prev(), {1,1,1}, shape);
    return u;
}
